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
from typing import Any

import torch

from nncf.common.utils.api_marker import api
from nncf.torch.dynamic_graph.context import forward_nncf_trace
from nncf.torch.nested_objects_traversal import objwalk
from nncf.torch.utils import is_tensor
from nncf.torch.utils import is_traced_tensor


@api(canonical_alias="nncf.torch.nncf_model_input")
def nncf_model_input(tensor: "torch.Tensor"):
    return tensor


@api(canonical_alias="nncf.torch.nncf_model_output")
def nncf_model_output(tensor: "torch.Tensor"):
    return tensor


def wrap_nncf_model_inputs_with_objwalk(model_args, model_kwargs):
    model_args = objwalk(model_args, is_tensor, nncf_model_input)
    model_kwargs = objwalk(model_kwargs, is_tensor, nncf_model_input)
    return model_args, model_kwargs


def wrap_nncf_model_outputs_with_objwalk(model_outputs):
    model_outputs = objwalk(model_outputs, is_traced_tensor, nncf_model_output)
    return model_outputs


def replicate_same_tensors(obj: Any) -> Any:
    """
    Required to handle the situation when multiple references to one and the
    same tensor are present in the input. If tensor replication is not done, then
    at runtime one and the same tensor could be wrapped by input/output wrappers twice,
    which will disrupt the traced graph structure and possibly hook calls.
    """
    observed_tensor_object_ids: set[int] = set()

    def replicate_fn(tensor: torch.Tensor) -> torch.Tensor:
        tensor_object_id = id(tensor)
        if tensor_object_id in observed_tensor_object_ids:
            with forward_nncf_trace():
                return tensor.clone()
        observed_tensor_object_ids.add(tensor_object_id)
        return tensor

    obj = objwalk(obj, is_tensor, replicate_fn)
    return obj
