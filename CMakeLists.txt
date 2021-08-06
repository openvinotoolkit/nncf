cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project (CMAKE_EXTENSIONS_BUILD)


find_package (PythonInterp 3.7 REQUIRED)
find_program (PYTEST pytest)

if(NOT PYTEST)
    message(FATAL_ERROR "No pytest binary find")
endif()

add_custom_command(
    COMMAND ${PYTEST} -s tests/torch/test_extensions_build.py
)

add_custom_target(EXTENSIONS_BUILD ALL
    DEPENDS tests/torch/test_extensions_build.py
    SOURCES tests/torch/test_extensions_build.py
)

add_test(NAME run_tests
    COMMAND ${PYTEST} -s tests/torch/test_extensions_build.py
)


