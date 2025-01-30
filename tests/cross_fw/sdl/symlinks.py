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
import abc
from pathlib import Path
from typing import Any, Callable

import pytest

import nncf
from nncf.common.utils.os import is_windows


def check_symlinks_are_not_followed(tmp_path: Path, file_opening_entrypoint: Callable[[str], Any]):
    if is_windows():
        pytest.skip("Symlinks are not supported on Windows")
    symlink_path = tmp_path / "symlink"
    real_file = tmp_path / "real_file"
    real_file.touch(exist_ok=True)
    symlink_path.symlink_to(real_file)
    with pytest.raises(nncf.ValidationError, match="is a symbolic link, aborting"):
        file_opening_entrypoint(str(symlink_path))


class FileOpeningEntrypointProvider(abc.ABC):
    @abc.abstractmethod
    def get_entrypoint(self) -> Callable[[str], Any]:
        pass
