import enum
import os
import signal
import textwrap
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Callable

import torch
from torch.utils.cpp_extension import _get_build_directory

from nncf.common.logging import nncf_logger
from nncf.common.logging.logger import extension_is_loading_info_log
from nncf.common.utils.api_marker import api
from nncf.common.utils.registry import Registry

EXTENSIONS = Registry("extensions")


class ExtensionsType(enum.Enum):
    CPU = 0
    CUDA = 1


def get_build_directory_for_extension(name: str) -> Path:
    build_dir = Path(_get_build_directory("nncf/" + name, verbose=False)) / torch.__version__
    if not build_dir.exists():
        nncf_logger.debug(f"Creating build directory: {str(build_dir)}")
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


class ExtensionLoaderTimeoutException(Exception):
    """Raises an exception if it takes too long time to load the extension"""


NNCF_TIME_LIMIT_TO_LOAD_EXTENSION = "NNCF_TIME_LIMIT_TO_LOAD_EXTENSION"


class ExtensionNamespace:
    """
    Provides lazy loading of the underlying extension, i.e. on the first request of a function from the extension.
    """

    def __init__(self, loader: ExtensionLoader):
        """
        :param loader: The extension loader.
        """
        self._loaded_namespace = None
        self._loader = loader

    def get(self, fn_name: str) -> Callable:
        """
        Returns the callable object corresponding to a function from the extension, loading the extension first if
        necessary.
        :param fn_name: A function name as normally accessible from the object return by torch extension's native
          .load() method.
        :return: A callable object corresponding to the requested function.
        """
        if self._loaded_namespace is None:
            time_limit = int(os.environ.get(NNCF_TIME_LIMIT_TO_LOAD_EXTENSION, 60))

            def extension_loader_timeout_handler(signum, frame):
                # pylint: disable=line-too-long
                msg = textwrap.dedent(
                    f"""\
                    The extension load function failed to execute within {time_limit} seconds.
                    To resolve this issue, run the following command:
                        rm -rf {self._loader.get_build_dir()}
                    For a machine with poor performance, you may try increasing the time limit by setting the environment variable:
                        {NNCF_TIME_LIMIT_TO_LOAD_EXTENSION}=180
                    More information about reasons read on https://github.com/openvinotoolkit/nncf/blob/develop/docs/FAQ.md#importing-anything-from-nncftorch-hangs
                    """
                )
                raise ExtensionLoaderTimeoutException(msg)

            with extension_is_loading_info_log(self._loader.name()):
                signal.signal(signal.SIGALRM, extension_loader_timeout_handler)
                signal.alarm(time_limit)
                try:
                    self._loaded_namespace = self._loader.load()
                finally:
                    signal.alarm(0)

        return getattr(self._loaded_namespace, fn_name)


def _force_build_extensions(ext_type: ExtensionsType):
    for class_type in EXTENSIONS.registry_dict.values():
        if class_type.extension_type() != ext_type:
            continue
        class_type.load()


@api(canonical_alias="nncf.torch.force_build_cpu_extensions")
def force_build_cpu_extensions():
    _force_build_extensions(ExtensionsType.CPU)


@api(canonical_alias="nncf.torch.force_build_cuda_extensions")
def force_build_cuda_extensions():
    _force_build_extensions(ExtensionsType.CUDA)


class CudaNotAvailableStub:
    def __getattr__(self, item):
        raise RuntimeError(
            f"CUDA is not available on this machine. Check that the machine has a GPU and a proper "
            f"driver supporting CUDA {torch.version.cuda} is installed."
        )
