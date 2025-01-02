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

from typing import Tuple

import openvino as ov
from packaging import version


def get_openvino_major_minor_version() -> Tuple[int]:
    ov_version = ov.__version__
    pos = ov_version.find("-")
    if pos != -1:
        ov_version = ov_version[:pos]

    ov_version = version.parse(ov_version).base_version
    return tuple(map(int, ov_version.split(".")[:2]))


def get_openvino_version() -> str:
    major_version, minor_version = get_openvino_major_minor_version()
    return f"{major_version}.{minor_version}"
