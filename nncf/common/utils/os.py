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
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, BinaryIO, Iterator, TextIO, Union

import psutil

import nncf


def fail_if_symlink(file: Path) -> None:
    if file.is_symlink():
        raise nncf.ValidationError("File {} is a symbolic link, aborting.".format(str(file)))


@contextmanager
def safe_open(file: Path, *args, **kwargs) -> Iterator[Union[TextIO, BinaryIO, IO[Any]]]:  # type: ignore
    """
    Safe function to open file and return a stream.

    For security reasons, should not follow symlinks. Use .resolve() on any Path
    objects before passing them here.

    :param file: The path to the file.
    :return: A file object.
    """
    fail_if_symlink(file)
    with open(str(file), *args, **kwargs) as f:
        yield f


def is_windows() -> bool:
    return "win32" in sys.platform


def is_linux() -> bool:
    return "linux" in sys.platform


def is_macos() -> bool:
    return "darwin" in sys.platform


def get_available_cpu_count(logical: bool = True) -> int:
    """
    Return the number of CPUs in the system.

    :param logical: If False return the number of physical cores only (e.g. hyper thread CPUs are excluded),
      otherwise number of logical cores. Defaults, True.
    :return: Number of CPU.
    """
    try:
        num_cpu = psutil.cpu_count(logical=logical)
        return num_cpu if num_cpu is not None else 1
    except Exception:
        return 1


def get_available_memory_amount() -> float:
    """
    :return: Available memory amount (bytes)
    """
    try:
        return psutil.virtual_memory()[1]
    except Exception:
        return 0
