cmake_minimum_required (VERSION 3.9 FATAL_ERROR)
project (tensor_funcs.o LANGUAGES CXX)

find_package (PythonInterp 3.7 REQUIRED)

execute_process (
    COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print (get_python_lib())"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE
    )

add_executable (tensor_funcs.o tensor_funcs.cpp)

list (APPEND cflags
    "-MMD"
    "-MF"
    "tensor_funcs.o.d"
    "-DTORCH_EXTENSION_NAME=binarized_functions_cpu"
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

list (APPEND ldflags
    "-shared"
    "-L${PYTHON_SITE_PACKAGES}/torch/lib"
    "-lc10"
    "-ltorch_cpu"
    "-ltorch"
    "-ltorch_python"
    )

target_compile_options (tensor_funcs.o
    PRIVATE
        ${cflags}
    )

target_link_libraries (tensor_funcs.o
    PRIVATE
        ${ldflags}
    )
