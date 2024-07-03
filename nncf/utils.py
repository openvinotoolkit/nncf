# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib

_openvino_available = importlib.util.find_spec("openvino") is not None
_openvino_version = "N/A"
if _openvino_available:
    try:
        from openvino.runtime import get_version

        version = get_version()
        # avoid invalid format
        if "-" in version:
            ov_major_version, dev_info = version.split("-", 1)
            commit_id = dev_info.split("-")[0]
            version = f"{ov_major_version}-{commit_id}"
        _openvino_version = version
    except ImportError:
        _openvino_available = False


def is_openvino_available():
    return _openvino_available
