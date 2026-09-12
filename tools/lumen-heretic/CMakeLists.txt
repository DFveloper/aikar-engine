set(TARGET lumen-heretic)

add_executable(${TARGET}
    dataset.cpp
    dataset.h
    direction.cpp
    direction.h
    evaluator.cpp
    evaluator.h
    main.cpp
    scorer.cpp
    scorer.h
    surgery.cpp
    surgery.h
)

target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/vendor)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (BUILD_TESTING)
    add_executable(test-lumen-heretic test-surgery.cpp surgery.cpp dataset.cpp)
    target_link_libraries(test-lumen-heretic PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
    target_include_directories(test-lumen-heretic PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/vendor)
    target_compile_features(test-lumen-heretic PRIVATE cxx_std_17)
    add_test(NAME test-lumen-heretic COMMAND $<TARGET_FILE:test-lumen-heretic>)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
