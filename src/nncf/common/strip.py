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


from typing import Any, Optional, TypeVar

import nncf
from nncf.common.utils.api_marker import api
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.parameters import StripFormat
from nncf.telemetry.decorator import tracked_function
from nncf.telemetry.events import MODEL_BASED_CATEGORY
from nncf.telemetry.extractors import FunctionCallTelemetryExtractor

TModel = TypeVar("TModel")


@api(canonical_alias="nncf.strip")
@tracked_function(category=MODEL_BASED_CATEGORY, extractors=[FunctionCallTelemetryExtractor("nncf.strip")])
def strip(
    model: TModel,
    do_copy: bool = True,
    strip_format: StripFormat = StripFormat.NATIVE,
    example_input: Optional[Any] = None,
) -> TModel:
    """
    Removes auxiliary layers and operations added during the compression process, resulting in a clean
    model ready for deployment. The functionality of the model object is still preserved as a compressed model.

    :param model: The compressed model.
    :param do_copy: If True (default), will return a copy of the currently associated model object. If False,
      will return the currently associated model object "stripped" in-place.
    :param strip format: Describes the format in which model is saved after strip.
    :param example_input: An example input tensor to be used for tracing the model.
    :return: The stripped model.
    """
    model_backend = get_backend(model)
    if model_backend == BackendType.TORCH:
        from nncf.torch.strip import strip as strip_pt

        return strip_pt(model, do_copy, strip_format, example_input)  # type: ignore

    msg = f"Method `strip` does not support {model_backend.value} backend."
    raise nncf.UnsupportedBackendError(msg)
