# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import os
import textwrap
import warnings
from abc import ABC
from abc import abstractmethod
from multiprocessing.context import TimeoutError as MPTimeoutError
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Callable

import torch
from torch.utils.cpp_extension import _get_build_directory

import nncf
from nncf.common.logging import nncf_logger
from nncf.common.utils.api_marker import api
from nncf.common.utils.registry import Registry

EXTENSIONS = Registry("extensions")

EXTENSION_LOAD_TIMEOUT_ENV_VAR = "NNCF_EXTENSION_LOAD_TIMEOUT"
DEFAULT_EXTENSION_LOAD_TIMEOUT = 60


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
    """Raised when the extension takes too long to load"""


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
            timeout = int(os.environ.get(EXTENSION_LOAD_TIMEOUT_ENV_VAR, DEFAULT_EXTENSION_LOAD_TIMEOUT))
            timeout = timeout if timeout > 0 else None
            nncf_logger.info(f"Compiling and loading torch extension: {self._loader.name()}...")
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="TORCH_CUDA_ARCH_LIST is not set")
                    pool = ThreadPool(processes=1)
                    async_result = pool.apply_async(self._loader.load)
                    self._loaded_namespace = async_result.get(timeout=timeout)
            except MPTimeoutError as error:
                msg = textwrap.dedent(
                    f"""\
                    The extension load function failed to execute within {timeout} seconds.
                    This may be due to leftover lock files from the PyTorch C++ extension build process.
                    If this is the case, running the following command should help:
                        rm -rf {self._loader.get_build_dir()}
                    For a machine with poor performance, you may try increasing the time limit by setting the environment variable:
                        {EXTENSION_LOAD_TIMEOUT_ENV_VAR}=180
                    Or disable timeout by set:
                        {EXTENSION_LOAD_TIMEOUT_ENV_VAR}=0
                    For more information, see FAQ entry at: https://github.com/openvinotoolkit/nncf/blob/develop/docs/FAQ.md#importing-anything-from-nncftorch-hangs
                    """  # noqa: E501
                )
                raise ExtensionLoaderTimeoutException(msg) from error
            nncf_logger.info(f"Finished loading torch extension: {self._loader.name()}")

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
        raise nncf.InstallationError(
            f"CUDA is not available on this machine. Check that the machine has a GPU and a proper "
            f"driver supporting CUDA {torch.version.cuda} is installed."
        )
