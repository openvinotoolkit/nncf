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

import re

import cpuinfo  # type: ignore

_IS_LNL_CPU = None


def is_lnl_cpu() -> bool:
    global _IS_LNL_CPU
    if _IS_LNL_CPU is None:
        _IS_LNL_CPU = re.search(r"Ultra \d 2\d{2}", cpuinfo.get_cpu_info()["brand_raw"]) is not None
    return _IS_LNL_CPU
