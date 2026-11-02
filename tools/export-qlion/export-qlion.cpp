#include "qlion-adapter.h"

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string base, output;
    std::vector<std::string> adapters;
    bool compatible = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--allow-compatible-base") {
            compatible = true;
        } else if ((arg == "-m" || arg == "--model" || arg == "-o" || arg == "--output" || arg == "--adapter") && i + 1 < argc) {
            const std::string value = argv[++i];
            if (arg == "-m" || arg == "--model") base = value;
            else if (arg == "-o" || arg == "--output") output = value;
            else adapters.push_back(value);
        } else {
            fprintf(stderr, "usage: llama-export-qlion -m BASE --adapter FILE [--adapter FILE ...] -o OUTPUT [--allow-compatible-base]\n");
            return arg == "--help" || arg == "-h" ? 0 : 1;
        }
    }
    if (base.empty() || output.empty() || adapters.empty()) {
        fprintf(stderr, "base, adapter and output are required; see --help\n");
        return 1;
    }
    try {
        common_qlion_merge(base, adapters, output, compatible);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
