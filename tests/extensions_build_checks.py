import sys
from .test_extensions_build import test_if_cuda_was_built_without_gpu

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise RuntimeError("Must be run with torch extension build directory")
    torch_build_dir = sys.argv[1]
    test_if_cuda_was_built_without_gpu(torch_build_dir)
