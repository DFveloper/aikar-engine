#include "scorer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>

namespace {

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
    return value;
}

}

heretic_prefix_scorer::heretic_prefix_scorer(const std::string & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("failed to open refusal-pattern file: " + path);
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        heretic_pattern pattern;
        const size_t tab = line.find('\t');
        if (tab != std::string::npos) {
            pattern.weight = std::stod(line.substr(0, tab));
            pattern.text = line.substr(tab + 1);
        } else {
            pattern.text = line;
        }
        pattern.text = lower(pattern.text);
        if (pattern.text.empty() || pattern.weight <= 0.0 || pattern.weight > 1.0) throw std::runtime_error("invalid refusal pattern: " + line);
        patterns.push_back(std::move(pattern));
    }
    if (patterns.empty()) throw std::runtime_error("refusal-pattern file is empty");
}

double heretic_prefix_scorer::score(const std::string & response) const {
    const std::string text = lower(response);
    double result = 0.0;
    for (const heretic_pattern & pattern : patterns) {
        if (text.find(pattern.text) != std::string::npos) result = std::max(result, pattern.weight);
    }
    return result;
}

size_t heretic_prefix_scorer::pattern_count() const {
    return patterns.size();
}

