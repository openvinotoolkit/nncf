import enum
import torch

import os

from nncf.common.utils.registry import Registry

EXTENSIONS = Registry('extensions')


class ExtensionsType(enum.Enum):
    CPU = 0
    CUDA = 1


def _force_build_extensions(ext_type):
    for class_type in EXTENSIONS.registry_dict.values():
        if class_type.extension_type() != ext_type:
            continue
        class_type.load()


def force_build_cpu_extensions():
    _force_build_extensions(ExtensionsType.CPU)


def force_build_cuda_extensions(arch_list=None):
    """
    arch_list can be one or more architectures, e.g. arch_list = ['3.5','5.2','6.0','6.1','7.0+PTX']
    See cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
    """
    if arch_list:
        saved_env = os.environ.get('TORCH_CUDA_ARCH_LIST', '')
        os.environ['TORCH_CUDA_ARCH_LIST'] = ';'.join(arch_list)
        _force_build_extensions(ExtensionsType.CUDA)
        os.environ['TORCH_CUDA_ARCH_LIST'] = saved_env
    else:
        _force_build_extensions(ExtensionsType.CUDA)

class CudaNotAvailableStub:
    def __getattr__(self, item):
        raise RuntimeError("CUDA is not available on this machine. Check that the machine has a GPU and a proper"
                           "driver supporting CUDA {} is installed".format(torch.version.cuda))
