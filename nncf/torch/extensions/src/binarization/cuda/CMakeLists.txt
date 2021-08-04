cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(binarized_functions_cuda.so LANGUAGES CXX CUDA)

find_package(PythonInterp 3.7 REQUIRED)

execute_process (
    COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print (get_python_lib())"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE
    )

add_executable(functions_cuda.o functions_cuda.cpp)
add_executable(functions_cuda_impl.cuda.o functions_cuda_impl.cu)

list(APPEND cflags
    "-MMD"
    "-MF"
    "functions_cuda.o.d"
    "-DTORCH_EXTENSION_NAME=binarized_functions_cuda"
    "-DTORCH_API_INCLUDE_EXTENSION_H"
    "-DPYBIND11_COMPILER_TYPE=\"_gcc\""
    "-DPYBIND11_STDLIB=\"_libstdcpp\""
    "-DPYBIND11_BUILD_ABI=\"_cxxabi1011\""
    "-I${PYTHON_SITE_PACKAGES}/nncf/torch/extensions/include"
    "-isystem${PYTHON_SITE_PACKAGES}/torch/include"
    "-isystem${PYTHON_SITE_PACKAGES}/torch/include/torch/csrc/api/include"
    "-isystem${PYTHON_SITE_PACKAGES}/torch/include/TH"
    "-isystem${PYTHON_SITE_PACKAGES}/torch/include/THC"
    "-isystem/localdisk/fk/venv/include/python3.7m"
    "-D_GLIBCXX_USE_CXX11_ABI=0"
    "-fPIC"
    "-std=c++14"
    )

list(APPEND cflags_cuda
    "-DTORCH_EXTENSION_NAME=binarized_functions_cuda"
    "-DTORCH_API_INCLUDE_EXTENSION_H"
    "-DPYBIND11_COMPILER_TYPE=\"_gcc\""
    "-DPYBIND11_STDLIB=\"_libstdcpp\""
    "-DPYBIND11_BUILD_ABI=\"_cxxabi1011\""
    "-I${PYTHON_SITE_PACKAGES}/nncf/torch/extensions/include"
    "-isystem${PYTHON_SITE_PACKAGES}/torch/include"
    "-isystem${PYTHON_SITE_PACKAGES}/torch/include/torch/csrc/api/include"
    "-isystem${PYTHON_SITE_PACKAGES}/torch/include/TH"
    "-isystem${PYTHON_SITE_PACKAGES}/torch/include/THC"
    "-isystem/localdisk/fk/venv/include/python3.7m"
    "-D_GLIBCXX_USE_CXX11_ABI=0"
     "-D__CUDA_NO_HALF_OPERATORS__ "
     "-D__CUDA_NO_HALF_CONVERSIONS__ "
     "-D__CUDA_NO_BFLOAT16_CONVERSIONS__ "
     "-D__CUDA_NO_HALF2_OPERATORS__ "
     "--expt-relaxed-constexpr "
     "-gencode=arch=compute_75,code=compute_75 "
     "-gencode=arch=compute_75,code=sm_75 "
     "--compiler-options '-fPIC' "
     "-std=c++14"
    )

list(APPEND ldflags
    "-shared"
    "-L${PYTHON_SITE_PACKAGES}/torch/lib"
    "-lc10"
    "-lc10_cuda"
    "-ltorch_cpu"
    "-ltorch_cuda"
    "-ltorch"
    "-ltorch_python"
    "-L/usr/local/cuda/lib64"
    "-lcudart"
    )

target_compile_options(functions_cuda.o
    PRIVATE
        ${cflags}
        )

target_compile_options(functions_cuda_impl.cuda.o
    PRIVATE
        ${cflags_cuda}
        )

target_link_libraries(binarized_functions_cuda.so
    PRIVATE
        functions_cuda.o
        functions_cuda_impl.cuda.o
        ${ldflags}
        )
