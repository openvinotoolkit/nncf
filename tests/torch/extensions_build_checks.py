import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise RuntimeError('Must be run with target extensions build mode')
    mode = sys.argv[1]
    if mode == 'cpu':
        # Do not remove - the import here is for testing purposes.
        # pylint: disable=wrong-import-position
        from nncf.torch import force_build_cpu_extensions

        # pylint: enable=wrong-import-position

        force_build_cpu_extensions()
    elif mode == 'cuda':
        # pylint: disable=wrong-import-position
        from nncf.torch import force_build_cuda_extensions

        # pylint: enable=wrong-import-position

        # Set CUDA Architecture
        # See cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
        os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5+PTX'
        force_build_cuda_extensions()
    else:
        raise RuntimeError('Invalid mode type!')
