cmake_minimum_required (VERSION 3.9 FATAL_ERROR)
project (quantized_functions_cpu.so LANGUAGES CXX)

find_package (PythonInterp 3.7 REQUIRED)
execute_process (
    COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print (get_python_lib())"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE
    )

add_library (tensor_funcs.o ../../common/cpu/tensor_funcs.cpp)

add_executable (quantized_functions_cpu.so functions_cpu.cpp)

list(APPEND cflags
    -MMD
    -MF
    functions_cpu.o.d
    -DTORCH_EXTENSION_NAME=quantized_functions_cuda
    -DTORCH_API_INCLUDE_EXTENSION_H
    -DPYBIND11_COMPILER_TYPE=\"_gcc\"
    -DPYBIND11_STDLIB=\"_libstdcpp\"
    -DPYBIND11_BUILD_ABI=\"_cxxabi1011\"
    -I../../../../include
    -isystem${PYTHON_SITE_PACKAGES}/torch/include
    -isystem${PYTHON_SITE_PACKAGES}/torch/include/torch/csrc/api/include
    -isystem${PYTHON_SITE_PACKAGES}/torch/include/TH
    -isystem${PYTHON_SITE_PACKAGES}/torch/include/THC
    -isystem/localdisk/fk/venv/include/python3.7m
    -D_GLIBCXX_USE_CXX11_ABI=0
    -fPIC
    -std=c++14
    )

list(APPEND ldflags
    -shared
    -L${PYTHON_SITE_PACKAGES}/torch/lib
    -lc10
    -ltorch_cpu
    -ltorch
    -ltorch_python
    )

target_compile_options (tensor_funcs.o
    PRIVATE
        ${cflags}
    )

target_compile_options (quantized_functions_cpu.so
    PRIVATE
        ${cflags}
    )

target_link_libraries (quantized_functions_cpu.so
    PRIVATE
        tensor_funcs.o
        ${ldflags}
    )
