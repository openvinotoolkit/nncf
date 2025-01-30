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

import os
import sys
from pathlib import Path

import pytest

from nncf import ValidationError
from nncf.common.utils.os import fail_if_symlink
from nncf.common.utils.os import get_available_cpu_count
from nncf.common.utils.os import get_available_memory_amount
from nncf.common.utils.os import is_linux
from nncf.common.utils.os import is_windows
from nncf.common.utils.os import safe_open


@pytest.fixture
def setup_links(tmpdir):
    temp_file = tmpdir.join("example.txt")
    temp_file.write_text("example text", encoding="utf-8")
    file_not_symlink = Path(temp_file.strpath)

    symlink_dir = tmpdir.mkdir("symlink_dir")
    file_symlink = Path(symlink_dir.strpath + "/symlink")
    os.symlink(temp_file.strpath, file_symlink)

    return {"file_symlink": file_symlink, "file_not_symlink": file_not_symlink}


def test_fail_if_symlink_not_symlink(setup_links):
    fail_if_symlink(setup_links["file_not_symlink"])


def test_fail_if_symlink_is_symlink(setup_links):
    with pytest.raises(ValidationError):
        fail_if_symlink(setup_links["file_symlink"])


def test_safe_open_not_symlink(setup_links):
    with safe_open(setup_links["file_not_symlink"], "r") as file_stream:
        content = file_stream.read()
        assert content == "example text"


def test_safe_open_is_symlink(setup_links):
    with pytest.raises(ValidationError):
        with safe_open(setup_links["file_symlink"], "r"):
            pass


def test_is_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert is_windows()


def test_is_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert is_linux()


@pytest.fixture
def mock_cpu_count(mocker):
    return mocker.patch("psutil.cpu_count")


def test_get_available_cpu_count_logical(mock_cpu_count):
    mock_cpu_count.return_value = 4
    result = get_available_cpu_count()
    assert result == 4


def test_get_available_cpu_count_physical(mock_cpu_count):
    mock_cpu_count.return_value = 2
    result = get_available_cpu_count(logical=False)
    assert result == 2


def test_get_available_cpu_count_none(mock_cpu_count):
    mock_cpu_count.return_value = None
    result = get_available_cpu_count()
    assert result == 1


def test_get_available_cpu_count_exception(mock_cpu_count):
    mock_cpu_count.side_effect = Exception("Error fetching CPU count")
    result = get_available_cpu_count()
    assert result == 1


@pytest.fixture
def mock_virtual_memory(mocker):
    return mocker.patch("psutil.virtual_memory")


def test_get_available_memory_amount(mock_virtual_memory):
    mock_virtual_memory.return_value = (
        1000000000,
        500000000,
        50.0,
        500000000,
        500000000,
    )
    result = get_available_memory_amount()
    assert result == 500000000


def test_get_available_memory_amount_exception(mock_virtual_memory):
    mock_virtual_memory.side_effect = Exception("Error fetching virtual memory")
    result = get_available_memory_amount()
    assert result == 0
