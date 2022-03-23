import enum
from pathlib import Path

import torch

from abc import ABC, abstractmethod

from torch.utils.cpp_extension import _get_build_directory

from nncf.common.utils.registry import Registry
from nncf.common.utils.logger import logger as nncf_logger

EXTENSIONS = Registry('extensions')


class ExtensionsType(enum.Enum):
    CPU = 0
    CUDA = 1

def get_build_directory_for_extension(name: str) -> Path:
    build_dir = Path(_get_build_directory('nncf/' + name, verbose=False)) / torch.__version__
    if not build_dir.exists():
        nncf_logger.debug("Creating build directory: {}".format(str(build_dir)))
        build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir


class ExtensionLoader(ABC):
    @classmethod
    @abstractmethod
    def extension_type(cls):
        pass

    @classmethod
    @abstractmethod
    def load(cls):
        pass

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        pass

    @classmethod
    def get_build_dir(cls) -> str:
        return str(get_build_directory_for_extension(cls.name()))


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
        raise RuntimeError(f"CUDA is not available on this machine. Check that the machine has a GPU and a proper"
                           f"driver supporting CUDA {torch.version.cuda} is installed")
