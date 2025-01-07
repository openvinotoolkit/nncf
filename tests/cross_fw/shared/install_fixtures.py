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
from typing import Set

import pytest

from tests.cross_fw.shared.helpers import create_venv_with_nncf


@pytest.fixture(scope="function")
def tmp_venv_with_nncf(tmp_path, package_type: str, venv_type: str, extras: Set[str]):
    venv_path = create_venv_with_nncf(tmp_path, package_type, venv_type, backends=extras)
    return venv_path
