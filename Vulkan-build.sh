rm -rf build/
cmake -B build -DGGML_NATIVE=ON -DGGML_VULKAN=ON
cmake --build build --config Release -j16
