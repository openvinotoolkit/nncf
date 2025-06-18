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


from copy import deepcopy
from typing import Any, Optional, TypeVar

from torch import nn

import nncf
from nncf.common.check_features import is_torch_tracing_by_patching
from nncf.parameters import StripFormat

TModel = TypeVar("TModel", bound=nn.Module)


def strip(
    model: TModel,
    do_copy: bool = True,
    strip_format: StripFormat = StripFormat.NATIVE,
    example_input: Optional[Any] = None,
) -> TModel:
    """
    Removes auxiliary layers and operations added during the compression process, resulting in a clean
    model ready for deployment. The functionality of the model object is still preserved as a compressed model.

    :param do_copy: If True (default), will return a copy of the currently associated model object. If False,
        will return the currently associated model object "stripped" in-place.
    :param strip format: Describes the format in which model is saved after strip.
    :param example_input: An example input tensor to be used for tracing the model.
    :return: The stripped model.
    """
    if is_torch_tracing_by_patching():
        return model.nncf.strip(do_copy, strip_format)

    from nncf.torch.function_hook.strip import strip_quantized_model

    if example_input is None:
        msg = "Required example_input for strip model."
        raise nncf.InternalError(msg)
    model = deepcopy(model) if do_copy else model
    return strip_quantized_model(model, example_input, strip_format)
