"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from torch.nn import Module


def get_module_by_scope(model: Module, scope: 'Scope') -> Module:
    get_nncf_wrapped_model = getattr(model, 'get_nncf_wrapped_model', None)
    module = get_nncf_wrapped_model() if get_nncf_wrapped_model else model
    for scope_element in scope[1:]:  # omit first scope element which corresponds to base module
        if scope_element.calling_field_name is None:
            # The module used is being created in-place every time and never stored in the model,
            # happens for nn.Softmax in BERT implementations.
            return None
        # pylint: disable=protected-access
        next_module = module._modules.get(scope_element.calling_field_name)
        if next_module is None:
            raise RuntimeError("Could not find a {} module member in {} module of scope {} during node search"
                               .format(scope_element.calling_field_name,
                                       scope_element.calling_module_class_name,
                                       str(scope)))
        module = next_module
    return module
