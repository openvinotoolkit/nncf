# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from typing import Optional

NNCF_TF_CATEGORY = "nncf_tf"
NNCF_PT_CATEGORY = "nncf_pt"
NNCF_ONNX_CATEGORY = "nncf_onnx"
NNCF_OV_CATEGORY = "nncf_ov"

CURRENT_CATEGORY = None


def _set_current_category(category: str):
    global CURRENT_CATEGORY
    CURRENT_CATEGORY = category


def get_current_category() -> Optional[str]:
    return CURRENT_CATEGORY


@contextmanager
def telemetry_category(category: str) -> str:
    previous_category = get_current_category()
    _set_current_category(category)
    yield category
    _set_current_category(previous_category)
