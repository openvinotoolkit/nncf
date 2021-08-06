
set (CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set (CUDA_INCLUDE_DIRS /usr/local/cuda/include/ ${CUDA_INCLUDE_DIRS})

project (CMAKE_EXTENSIONS_BUILD)

find_package (PythonInterp 3.7 REQUIRED)



add_custom_command(
    OUTPUT venv.stamp
    DEPENDS venv requirements.txt
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/venv/bin/pip install -r requirements.txt
)

# Build command line to run py.test.
set(PYTEST
    ${CMAKE_CURRENT_BINARY_DIR}/venv/bin/python2
    ${CMAKE_CURRENT_BINARY_DIR}/venv/bin/py.test
)

add_custom_target(Tests ALL
    DEPENDS venv.stamp
    SOURCES requirements.txt
)

add_test(NAME run_tests
    COMMAND ${PYTEST} ${CMAKE_CURRENT_SOURCE_DIR}/test_sample.py
)


