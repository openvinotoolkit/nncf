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

from typing import TypeVar

import nncf
from nncf.common.factory import build_graph
from nncf.common.utils.api_marker import api
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend

TModel = TypeVar("TModel")


@api(canonical_alias="nncf.repack_weights")
def repack_weights(
    model: TModel,
) -> TModel:
    """
    Looking for 4 and 8 bit weights in OV model and repack them if maximal absolute value corresponds
    to the supported type with lower bits.

    :param model: A model to be repacked.
    :type model: ov.Model
    :return: The non-trainable model with repacked weights or the same model.
    """
    backend = get_backend(model)

    if backend != BackendType.OPENVINO:
        msg = f"Unsupported type of backend: {backend}"
        raise nncf.UnsupportedBackendError(msg)

    from nncf.openvino.graph.model_utils import remove_friendly_name_duplicates

    model = remove_friendly_name_duplicates(model)
    graph = build_graph(model)

    from nncf.quantization.algorithms.weight_compression.openvino_backend import repack_weights as repack_weights_impl

    return repack_weights_impl(model, graph)  # type: ignore[no-any-return]
