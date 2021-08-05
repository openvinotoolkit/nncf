cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

set (CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set (CUDA_INCLUDE_DIRS /usr/local/cuda/include/ ${CUDA_INCLUDE_DIRS})

project(quantized_functions_cuda.so LANGUAGES CXX CUDA)

find_package(PythonInterp 3.7 REQUIRED)

execute_process (
    COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print (get_python_lib())"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE
    )

add_library(functions_cuda.o
    STATIC
    functions_cuda.cpp)

list(APPEND cflags
    -MMD
    -MF
    functions_cuda.o.d
    -DTORCH_EXTENSION_NAME=quantized_functions_cuda
    -DTORCH_API_INCLUDE_EXTENSION_H
    -DPYBIND11_COMPILER_TYPE=\"_gcc\"
    -DPYBIND11_STDLIB=\"_libstdcpp\"
    -DPYBIND11_BUILD_ABI=\"_cxxabi1011\"
    -I${PYTHON_SITE_PACKAGES}/nncf/torch/extensions/include
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
    -lc10_cuda
    -ltorch_cpu
    -ltorch_cuda
    -ltorch
    -ltorch_python
    -L/usr/local/cuda/lib64
    -lcudart
    )

target_compile_options(functions_cuda.o
    PRIVATE
        ${cflags}
        )

target_compile_features(functions_cuda.o PUBLIC cxx_std_14)
set_target_properties( functions_cuda.o
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(quantized_functions_cuda.so functions_cuda_impl.cu)

set_property(TARGET quantized_functions_cuda.so
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(quantized_functions_cuda.so
    PRIVATE
    -DTORCH_EXTENSION_NAME=quantized_functions_cuda
    -DTORCH_API_INCLUDE_EXTENSION_H
    -DPYBIND11_COMPILER_TYPE=\"_gcc\"
    -DPYBIND11_STDLIB=\"_libstdcpp\"
    -DPYBIND11_BUILD_ABI=\"_cxxabi1011\"
    -I${PYTHON_SITE_PACKAGES}/nncf/torch/extensions/include
    --system-include ${PYTHON_SITE_PACKAGES}/torch/include,${PYTHON_SITE_PACKAGES}/torch/include/torch/csrc/api/include,${PYTHON_SITE_PACKAGES}/torch/include/TH,${PYTHON_SITE_PACKAGES}/torch/include/THC,/localdisk/fk/venv/include/python3.7m
    -D_GLIBCXX_USE_CXX11_ABI=0
    -D__CUDA_NO_HALF_OPERATORS__
    -D__CUDA_NO_HALF_CONVERSIONS__
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__
    -D__CUDA_NO_HALF2_OPERATORS__
    --expt-relaxed-constexpr
    -gencode=arch=compute_75,code=compute_75
    --compiler-options '-fPIC'
    -std=c++14
)

target_link_libraries(quantized_functions_cuda.so
    PRIVATE
        functions_cuda.o
        ${ldflags}
        )
