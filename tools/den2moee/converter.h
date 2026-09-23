#pragma once

#include "config.h"

#include <string>

namespace den2moee {

void convert_dense_gemma4(const std::string & model_path, const std::string & calibration_path,
                          const std::string & output_path, const Options & options);

} // namespace den2moee
