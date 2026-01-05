# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

import nncf


def test_nncf_version():
    # Validate format of version
    # Needed while src/custom_version.py exists
    version_pattern = r"^\d\.\d{1,2}\.\d{1,2}$"  # e.g., "3.0.0"
    assert re.match(version_pattern, nncf.__version__), (
        f"NNCF version '{nncf.__version__}' does not match the expected pattern."
    )
