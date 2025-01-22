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

from typing import Optional

import torch.nn

import nncf
from nncf.torch.dynamic_graph.scope import Scope


def get_module_by_scope(model: torch.nn.Module, scope: Scope) -> Optional[torch.nn.Module]:
    curr_module = model
    for scope_element in scope[1:]:  # omit first scope element which corresponds to base module
        if scope_element.calling_field_name is None:
            # The module used is being created in-place every time and never stored in the model,
            # happens for nn.Softmax in BERT implementations.
            return None

        next_module = curr_module._modules.get(scope_element.calling_field_name)
        if next_module is None:
            raise nncf.InternalError(
                "Could not find a {} module member in {} module of scope {} during node search".format(
                    scope_element.calling_field_name, scope_element.calling_module_class_name, str(scope)
                )
            )
        curr_module = next_module
    return curr_module
