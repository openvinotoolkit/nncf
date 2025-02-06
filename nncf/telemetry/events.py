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

from contextlib import contextmanager
from typing import Generator, Optional, TypeVar

from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend

# General categories
NNCF_CATEGORY = "nncf"

# Backend categories
NNCF_TF_CATEGORY = "nncf_tf"
NNCF_PT_CATEGORY = "nncf_pt"
NNCF_PT_FX_CATEGORY = "nncf_pt_fx"
NNCF_ONNX_CATEGORY = "nncf_onnx"
NNCF_OV_CATEGORY = "nncf_ov"

# Dynamic categories
MODEL_BASED_CATEGORY = "model_based"

CURRENT_CATEGORY: Optional[str] = None

TModel = TypeVar("TModel")


def _set_current_category(category: Optional[str]) -> None:
    global CURRENT_CATEGORY
    CURRENT_CATEGORY = category


def get_current_category() -> Optional[str]:
    return CURRENT_CATEGORY


def get_model_based_category(model: TModel) -> str:
    category_by_backend = {
        BackendType.ONNX: NNCF_ONNX_CATEGORY,
        BackendType.OPENVINO: NNCF_OV_CATEGORY,
        BackendType.TORCH: NNCF_PT_CATEGORY,
        BackendType.TENSORFLOW: NNCF_TF_CATEGORY,
        BackendType.TORCH_FX: NNCF_PT_FX_CATEGORY,
    }
    category = None
    if model is not None:
        model_backend = get_backend(model)
        category = category_by_backend[model_backend]

    return category


@contextmanager
def telemetry_category(category: Optional[str]) -> Generator[Optional[str], None, None]:
    previous_category = get_current_category()
    _set_current_category(category)
    yield category
    _set_current_category(previous_category)
