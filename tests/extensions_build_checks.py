import os
import sys
import pathlib

if len(sys.argv) != 2:
    raise RuntimeError("Must be run with torch extension build directory")
torch_build_dir = sys.argv[1]

torch_ext_dir = pathlib.Path(torch_build_dir)
assert not torch_ext_dir.exists()

from nncf import force_build_cuda_extensions

cpu_ext_dir = (torch_ext_dir / 'quantized_functions_cpu')
assert cpu_ext_dir.exists()
cpu_ext_so = (cpu_ext_dir / 'quantized_functions_cpu.so')
assert cpu_ext_so.exists()

cuda_ext_dir = (torch_ext_dir / 'quantized_functions_cuda')
assert not cuda_ext_dir.exists()
cuda_ext_so = (cuda_ext_dir / 'quantized_functions_cuda.so')
assert not cuda_ext_so.exists()

# Set CUDA Architecture
# See cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5+PTX'
force_build_cuda_extensions()

cuda_ext_dir = (torch_ext_dir / 'quantized_functions_cuda')
assert cuda_ext_dir.exists()
cuda_ext_so = (cuda_ext_dir / 'quantized_functions_cuda.so')
assert cuda_ext_so.exists()
