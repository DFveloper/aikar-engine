cmake_minimum_required(VERSION 3.16)

project(aikar-bench LANGUAGES CXX)

if(NOT UNIX OR APPLE)
    return()
endif()

add_executable(aikar-bench
    aikar-bench.cpp
)

target_compile_features(aikar-bench PRIVATE cxx_std_17)

target_compile_options(aikar-bench PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
        -O3
        -Wall
        -Wextra
        -Wpedantic
    >
)

find_package(Threads REQUIRED)

target_link_libraries(aikar-bench PRIVATE
    Threads::Threads
)

# Keep runtime binary in the normal project binary directory if the parent
# project already defines CMAKE_RUNTIME_OUTPUT_DIRECTORY.
if(NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set_target_properties(aikar-bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

install(
    TARGETS aikar-bench
    RUNTIME DESTINATION bin
)
