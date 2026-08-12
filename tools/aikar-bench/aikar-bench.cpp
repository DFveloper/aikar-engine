// aikar-bench.cpp
// Linux C++17, zero external dependencies.
//
// Build:
//   g++ -O3 -std=c++17 -pthread aikar-bench.cpp -o aikar-bench
//
// Fully static (recommended with musl):
//   musl-g++ -O3 -std=c++17 -pthread -static aikar-bench.cpp -o aikar-bench
//
// Example:
//   ./aikar-bench -h 127.0.0.1 -p 11435 -n 4 -t 1024

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Config {
    std::string host = "127.0.0.1";
    int port = 11435;
    int concurrency = 4;
    int max_tokens = 512;

    std::string model = "default";

    std::string prompt =
        "Explain speculative decoding in large language model inference "
        "in technical detail.";

    double temperature = 0.0;

    bool show_output = false;
};

struct Result {
    int id = 0;
    bool ok = false;

    double ttft = -1.0;
    double latency = 0.0;

    long prompt_tokens = 0;
    long completion_tokens = 0;

    std::string output;
    std::string error;

    double decode_tps() const {
        if (ttft < 0.0)
            return 0.0;

        double gen = latency - ttft;

        if (gen <= 0.0)
            return 0.0;

        return static_cast<double>(completion_tokens) / gen;
    }
};

class Barrier {
public:
    explicit Barrier(int count)
        : target(count) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex);

        arrived++;

        if (arrived == target) {
            ready = true;
            cv.notify_all();
        } else {
            cv.wait(lock, [&]() {
                return ready;
            });
        }
    }

private:
    int target;
    int arrived = 0;

    bool ready = false;

    std::mutex mutex;
    std::condition_variable cv;
};

static double seconds_between(
    const Clock::time_point& a,
    const Clock::time_point& b
) {
    return std::chrono::duration<double>(b - a).count();
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 32);

    for (unsigned char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;

        case '\\':
            out += "\\\\";
            break;

        case '\b':
            out += "\\b";
            break;

        case '\f':
            out += "\\f";
            break;

        case '\n':
            out += "\\n";
            break;

        case '\r':
            out += "\\r";
            break;

        case '\t':
            out += "\\t";
            break;

        default:
            if (c < 0x20) {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += static_cast<char>(c);
            }
        }
    }

    return out;
}

static int connect_tcp(
    const std::string& host,
    int port,
    std::string& error
) {
    struct addrinfo hints {};
    struct addrinfo* result = nullptr;

    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(port);

    int ret = getaddrinfo(
        host.c_str(),
        port_str.c_str(),
        &hints,
        &result
    );

    if (ret != 0) {
        error = std::string("getaddrinfo: ") +
                gai_strerror(ret);
        return -1;
    }

    int fd = -1;

    for (
        struct addrinfo* rp = result;
        rp != nullptr;
        rp = rp->ai_next
    ) {
        fd = socket(
            rp->ai_family,
            rp->ai_socktype,
            rp->ai_protocol
        );

        if (fd < 0)
            continue;

        int flag = 1;

        setsockopt(
            fd,
            IPPROTO_TCP,
            TCP_NODELAY,
            &flag,
            sizeof(flag)
        );

        if (connect(
                fd,
                rp->ai_addr,
                rp->ai_addrlen
            ) == 0) {
            break;
        }

        close(fd);
        fd = -1;
    }

    freeaddrinfo(result);

    if (fd < 0) {
        error = "connect() failed";
    }

    return fd;
}

static bool send_all(
    int fd,
    const std::string& data,
    std::string& error
) {
    const char* ptr = data.data();
    size_t left = data.size();

    while (left > 0) {
        ssize_t n = send(
            fd,
            ptr,
            left,
            MSG_NOSIGNAL
        );

        if (n <= 0) {
            error = "send() failed";
            return false;
        }

        ptr += n;
        left -= static_cast<size_t>(n);
    }

    return true;
}

static long find_json_integer(
    const std::string& text,
    const std::string& key
) {
    std::string needle = "\"" + key + "\"";

    size_t pos = text.find(needle);

    if (pos == std::string::npos)
        return -1;

    pos = text.find(':', pos);

    if (pos == std::string::npos)
        return -1;

    pos++;

    while (
        pos < text.size() &&
        (
            text[pos] == ' ' ||
            text[pos] == '\t'
        )
    ) {
        pos++;
    }

    char* end = nullptr;

    long value = std::strtol(
        text.c_str() + pos,
        &end,
        10
    );

    if (end == text.c_str() + pos)
        return -1;

    return value;
}

// Minimal parser for:
//
// "content":"hello..."
//
// Enough for benchmark output / detecting first real content.
// This is intentionally not a complete JSON parser.
static bool extract_json_string_field(
    const std::string& json,
    const std::string& field,
    std::string& value
) {
    const std::string key = "\"" + field + "\"";

    size_t pos = json.find(key);

    if (pos == std::string::npos)
        return false;

    pos = json.find(':', pos + key.size());

    if (pos == std::string::npos)
        return false;

    ++pos;

    while (
        pos < json.size() &&
        (json[pos] == ' ' || json[pos] == '\t')
    ) {
        ++pos;
    }

    if (
        pos + 4 <= json.size() &&
        json.compare(pos, 4, "null") == 0
    ) {
        return false;
    }

    if (
        pos >= json.size() ||
        json[pos] != '"'
    ) {
        return false;
    }

    ++pos;

    std::string out;

    while (pos < json.size()) {
        char c = json[pos++];

        if (c == '"') {
            value = std::move(out);
            return !value.empty();
        }

        if (c == '\\' && pos < json.size()) {
            char e = json[pos++];

            switch (e) {
            case '"':
                out += '"';
                break;
            case '\\':
                out += '\\';
                break;
            case '/':
                out += '/';
                break;
            case 'b':
                out += '\b';
                break;
            case 'f':
                out += '\f';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;

            default:
                out += '\\';
                out += e;
                break;
            }
        } else {
            out += c;
        }
    }

    return false;
}


static bool extract_generated_text(
    const std::string& json,
    std::string& text
) {
    // Normal assistant output
    if (extract_json_string_field(
            json,
            "content",
            text
        )) {
        return true;
    }

    // llama.cpp / reasoning-compatible APIs
    if (extract_json_string_field(
            json,
            "reasoning_content",
            text
        )) {
        return true;
    }

    // Some OpenAI-compatible implementations
    if (extract_json_string_field(
            json,
            "reasoning",
            text
        )) {
        return true;
    }

    return false;
}

static std::string make_body(
    const Config& cfg
) {
    std::ostringstream ss;

    ss
        << "{"
        << "\"model\":\""
        << json_escape(cfg.model)
        << "\","

        << "\"messages\":[{"
        << "\"role\":\"user\","
        << "\"content\":\""
        << json_escape(cfg.prompt)
        << "\""
        << "}],"

        << "\"max_tokens\":"
        << cfg.max_tokens
        << ","

        << "\"temperature\":"
        << cfg.temperature
        << ","

        << "\"stream\":true,"
        << "\"stream_options\":{"
        << "\"include_usage\":true"
        << "}"

        << "}";

    return ss.str();
}

static std::string make_request(
    const Config& cfg,
    const std::string& body
) {
    std::ostringstream ss;

    ss
        << "POST /v1/chat/completions HTTP/1.1\r\n"

        << "Host: "
        << cfg.host
        << ":"
        << cfg.port
        << "\r\n"

        << "Content-Type: application/json\r\n"
        << "Accept: text/event-stream\r\n"

        << "Connection: close\r\n"

        << "Content-Length: "
        << body.size()
        << "\r\n"

        << "\r\n"

        << body;

    return ss.str();
}

static void parse_sse_line(
    const std::string& line,
    Result& result,
    const Clock::time_point& begin,
    bool& got_first_token
) {
    if (line.rfind("data:", 0) != 0)
        return;

    std::string payload = line.substr(5);

    while (
        !payload.empty() &&
        payload.front() == ' '
    ) {
        payload.erase(payload.begin());
    }

    if (
        payload == "[DONE]" ||
        payload.empty()
    ) {
        return;
    }

    long p = find_json_integer(
        payload,
        "prompt_tokens"
    );

    long c = find_json_integer(
        payload,
        "completion_tokens"
    );

    if (p >= 0)
        result.prompt_tokens = p;

    if (c >= 0)
        result.completion_tokens = c;

    std::string content;

    if (extract_generated_text(payload, content)) {
        auto now = Clock::now();

        if (!got_first_token) {
            result.ttft =
                seconds_between(begin, now);

            got_first_token = true;
        }

        result.output += content;
    }
}

static Result worker(
    int id,
    const Config& cfg,
    Barrier& connected_barrier,
    Barrier& fire_barrier
) {
    Result result;
    result.id = id;

    std::string error;

    //
    // Important:
    // establish TCP connection BEFORE benchmark fire barrier.
    //
    int fd = connect_tcp(
        cfg.host,
        cfg.port,
        error
    );

    if (fd < 0) {
        result.error = error;

        // Avoid deadlocking all other workers if one fails.
        //
        // For simplicity, connection failures are fatal.
        std::cerr
            << "Worker "
            << id
            << " failed before barrier: "
            << error
            << "\n";

        std::exit(2);
    }

    connected_barrier.arrive_and_wait();

    std::string body = make_body(cfg);
    std::string request = make_request(
        cfg,
        body
    );

    //
    // Second barrier:
    // TCP is already connected.
    //
    fire_barrier.arrive_and_wait();

    auto begin = Clock::now();

    if (!send_all(fd, request, error)) {
        close(fd);

        result.error = error;
        result.latency =
            seconds_between(
                begin,
                Clock::now()
            );

        return result;
    }

    std::string buffer;
    buffer.reserve(65536);

    bool headers_done = false;
    bool got_first_token = false;

    char recvbuf[32768];

    while (true) {
        ssize_t n = recv(
            fd,
            recvbuf,
            sizeof(recvbuf),
            0
        );

        if (n == 0)
            break;

        if (n < 0) {
            result.error = "recv() failed";
            break;
        }

        buffer.append(
            recvbuf,
            static_cast<size_t>(n)
        );

        if (!headers_done) {
            size_t h = buffer.find(
                "\r\n\r\n"
            );

            if (h == std::string::npos)
                continue;

            std::string headers =
                buffer.substr(0, h);

            // Basic HTTP error check.
            if (
                headers.find(" 200 ") ==
                std::string::npos
            ) {
                result.error =
                    "HTTP response was not 200:\n" +
                    headers;

                break;
            }

            buffer.erase(
                0,
                h + 4
            );

            headers_done = true;
        }

        //
        // Parse complete SSE lines.
        //
        while (true) {
            size_t eol = buffer.find('\n');

            if (eol == std::string::npos)
                break;

            std::string line =
                buffer.substr(0, eol);

            buffer.erase(
                0,
                eol + 1
            );

            if (
                !line.empty() &&
                line.back() == '\r'
            ) {
                line.pop_back();
            }

            parse_sse_line(
                line,
                result,
                begin,
                got_first_token
            );
        }
    }

    auto end = Clock::now();

    close(fd);

    result.latency =
        seconds_between(begin, end);

    result.ok =
        result.error.empty();

    return result;
}

static void usage(
    const char* argv0
) {
    std::cout
        << "Usage: "
        << argv0
        << " [options]\n\n"

        << "Options:\n"
        << "  -h HOST       Server host (default 127.0.0.1)\n"
        << "  -p PORT       Server port (default 11435)\n"
        << "  -n N          Concurrent requests (default 4)\n"
        << "  -t TOKENS     max_tokens (default 512)\n"
        << "  -m MODEL      Model name (default default)\n"
        << "  -P PROMPT     Prompt\n"
        << "  -T TEMP       Temperature (default 0)\n"
        << "  -o            Show generated output\n"
        << "  --help        Show help\n";
}

int main(
    int argc,
    char** argv
) {
    Config cfg;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        auto require = [&](const char* name) {
            if (i + 1 >= argc) {
                std::cerr
                    << "Missing argument for "
                    << name
                    << "\n";

                std::exit(1);
            }

            return std::string(argv[++i]);
        };

        if (arg == "-h") {
            cfg.host = require("-h");
        }
        else if (arg == "-p") {
            cfg.port = std::stoi(
                require("-p")
            );
        }
        else if (arg == "-n") {
            cfg.concurrency = std::stoi(
                require("-n")
            );
        }
        else if (arg == "-t") {
            cfg.max_tokens = std::stoi(
                require("-t")
            );
        }
        else if (arg == "-m") {
            cfg.model = require("-m");
        }
        else if (arg == "-P") {
            cfg.prompt = require("-P");
        }
        else if (arg == "-T") {
            cfg.temperature = std::stod(
                require("-T")
            );
        }
        else if (arg == "-o") {
            cfg.show_output = true;
        }
        else if (
            arg == "--help" ||
            arg == "-?"
        ) {
            usage(argv[0]);
            return 0;
        }
        else {
            std::cerr
                << "Unknown argument: "
                << arg
                << "\n";

            usage(argv[0]);
            return 1;
        }
    }

    if (cfg.concurrency <= 0) {
        std::cerr
            << "Concurrency must be > 0\n";

        return 1;
    }

    std::cout
        << "\n=== aikar-engine concurrent benchmark ===\n"

        << "Server       : http://"
        << cfg.host
        << ":"
        << cfg.port
        << "\n"

        << "Endpoint     : /v1/chat/completions\n"

        << "Concurrency  : "
        << cfg.concurrency
        << "\n"

        << "Max tokens   : "
        << cfg.max_tokens
        << "\n"

        << "Prompt bytes : "
        << cfg.prompt.size()
        << "\n\n";

    //
    // +1 for main thread.
    //
    // This makes benchmark wall clock start extremely close to the
    // actual simultaneous send point.
    //
    Barrier connected_barrier(
        cfg.concurrency + 1
    );

    Barrier fire_barrier(
        cfg.concurrency + 1
    );

    std::vector<Result> results(
        cfg.concurrency
    );

    std::vector<std::thread> threads;
    threads.reserve(cfg.concurrency);

    for (
        int i = 0;
        i < cfg.concurrency;
        i++
    ) {
        threads.emplace_back(
            [&, i]() {
                results[i] = worker(
                    i,
                    cfg,
                    connected_barrier,
                    fire_barrier
                );
            }
        );
    }

    //
    // Wait until EVERY TCP connection has been established.
    //
    connected_barrier.arrive_and_wait();

    std::cout
        << "All "
        << cfg.concurrency
        << " TCP connections ready. FIRE.\n\n";

    auto wall_begin = Clock::now();

    //
    // Simultaneously release every worker.
    //
    fire_barrier.arrive_and_wait();

    for (auto& t : threads)
        t.join();

    auto wall_end = Clock::now();

    double wall =
        seconds_between(
            wall_begin,
            wall_end
        );

    std::cout
        << std::right
        << std::setw(4)
        << "ID"

        << std::setw(7)
        << "OK"

        << std::setw(12)
        << "TTFT"

        << std::setw(12)
        << "Latency"

        << std::setw(10)
        << "Prompt"

        << std::setw(10)
        << "Output"

        << std::setw(12)
        << "TPS"

        << "\n";

    std::cout
        << std::string(67, '-')
        << "\n";

    int successful = 0;

    long total_prompt = 0;
    long total_output = 0;

    double ttft_sum = 0.0;
    int ttft_count = 0;

    double latency_sum = 0.0;
    double tps_sum = 0.0;

    double min_ttft = 1e100;
    double max_ttft = 0.0;

    for (const auto& r : results) {
        std::cout
            << std::setw(4)
            << r.id

            << std::setw(7)
            << (r.ok ? "yes" : "NO");

        if (r.ttft >= 0.0) {
            std::cout
                << std::setw(12)
                << std::fixed
                << std::setprecision(3)
                << r.ttft;
        } else {
            std::cout
                << std::setw(12)
                << "-";
        }

        std::cout
            << std::setw(12)
            << std::fixed
            << std::setprecision(3)
            << r.latency

            << std::setw(10)
            << r.prompt_tokens

            << std::setw(10)
            << r.completion_tokens

            << std::setw(12)
            << std::fixed
            << std::setprecision(2)
            << r.decode_tps()

            << "\n";

        if (!r.ok) {
            std::cout
                << "    ERROR: "
                << r.error
                << "\n";

            continue;
        }

        successful++;

        total_prompt += r.prompt_tokens;
        total_output += r.completion_tokens;

        latency_sum += r.latency;
        tps_sum += r.decode_tps();

        if (r.ttft >= 0.0) {
            ttft_sum += r.ttft;
            ttft_count++;

            min_ttft =
                std::min(
                    min_ttft,
                    r.ttft
                );

            max_ttft =
                std::max(
                    max_ttft,
                    r.ttft
                );
        }
    }

    std::cout
        << "\n=== Summary ===\n"

        << "Successful       : "
        << successful
        << "/"
        << cfg.concurrency
        << "\n"

        << "Wall time        : "
        << std::fixed
        << std::setprecision(3)
        << wall
        << " s\n"

        << "Prompt tokens    : "
        << total_prompt
        << "\n"

        << "Output tokens    : "
        << total_output
        << "\n";

    if (ttft_count > 0) {
        std::cout
            << "TTFT avg/min/max : "
            << std::fixed
            << std::setprecision(3)

            << (
                ttft_sum /
                ttft_count
            )

            << " / "
            << min_ttft

            << " / "
            << max_ttft

            << " s\n";
    }

    if (successful > 0) {
        std::cout
            << "Latency avg      : "
            << std::fixed
            << std::setprecision(3)

            << (
                latency_sum /
                successful
            )

            << " s\n"

            << "Individual TPS   : "
            << std::fixed
            << std::setprecision(2)

            << (
                tps_sum /
                successful
            )

            << " avg\n"

            << "Aggregate TPS    : "
            << std::fixed
            << std::setprecision(2)

            << (
                wall > 0.0
                ? total_output / wall
                : 0.0
            )

            << " tok/s\n"

            << "Requests/sec     : "
            << std::fixed
            << std::setprecision(3)

            << (
                wall > 0.0
                ? successful / wall
                : 0.0
            )

            << " req/s\n";
    }

    if (cfg.show_output) {
        for (const auto& r : results) {
            std::cout
                << "\n===== REQUEST "
                << r.id
                << " =====\n";

            if (r.ok)
                std::cout << r.output << "\n";
            else
                std::cout << r.error << "\n";
        }
    }

    return successful ==
                   cfg.concurrency
               ? 0
               : 2;
}
