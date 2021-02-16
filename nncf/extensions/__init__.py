import enum
import torch

from abc import ABC, abstractmethod
from nncf.common.utils.registry import Registry

EXTENSIONS = Registry('extensions')


class ExtensionsType(enum.Enum):
    CPU = 0
    CUDA = 1


class ExtensionLoader(ABC):
    @staticmethod
    @abstractmethod
    def extension_type():
        pass

    @staticmethod
    @abstractmethod
    def load():
        pass


def _force_build_extensions(ext_type: ExtensionsType):
    for class_type in EXTENSIONS.registry_dict.values():
        if class_type.extension_type() != ext_type:
            continue
        class_type.load()


def force_build_cpu_extensions():
    _force_build_extensions(ExtensionsType.CPU)


def force_build_cuda_extensions():
    _force_build_extensions(ExtensionsType.CUDA)


class CudaNotAvailableStub:
    def __getattr__(self, item):
        raise RuntimeError("CUDA is not available on this machine. Check that the machine has a GPU and a proper"
                           "driver supporting CUDA {} is installed".format(torch.version.cuda))
