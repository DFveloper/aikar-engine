set(TARGET llama-den2moee-convert)

find_library(DEN2MOEE_LAPACK_LIBRARY NAMES lapack)
find_library(DEN2MOEE_BLAS_LIBRARY NAMES blas)
find_package(OpenMP QUIET)

add_executable(${TARGET}
    calibration.cpp
    clustering.cpp
    config.cpp
    converter.cpp
    gguf_io.cpp
    main.cpp
    quantize.cpp
    router.cpp
    scoring.cpp
    self-test.cpp
    self-test.h
    svd.cpp
)

target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(OpenMP_CXX_FOUND)
    target_compile_definitions(${TARGET} PRIVATE DEN2MOEE_USE_OPENMP=1)
    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
endif()

if(DEN2MOEE_LAPACK_LIBRARY AND DEN2MOEE_BLAS_LIBRARY)
    target_compile_definitions(${TARGET} PRIVATE DEN2MOEE_USE_LAPACK=1)
    target_link_libraries(${TARGET} PRIVATE ${DEN2MOEE_LAPACK_LIBRARY} ${DEN2MOEE_BLAS_LIBRARY})
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
