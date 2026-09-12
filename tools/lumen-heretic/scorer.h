#pragma once

#include <string>
#include <vector>

struct heretic_pattern {
    std::string text;
    double weight = 1.0;
};

class heretic_prefix_scorer {
public:
    explicit heretic_prefix_scorer(const std::string & path);

    double score(const std::string & response) const;
    size_t pattern_count() const;

private:
    std::vector<heretic_pattern> patterns;
};

