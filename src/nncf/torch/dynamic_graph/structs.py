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


from enum import Enum


class NamespaceTarget(Enum):
    """
    NamespaceTarget stores modules from which patched operators were obtained.
    """

    TORCH_NN_FUNCTIONAL = "torch.nn.functional"
    TORCH_TENSOR = "torch.tensor"
    TORCH_NN_PARAMETER = "torch.nn.parameter"
    TORCH = "torch"
    ATEN = "aten"
    EXTERNAL = "external_function"


class PatchedOperatorInfo:
    def __init__(self, name: str, operator_namespace: NamespaceTarget, skip_trace: bool = False):
        """
        Information about patched operator.
        :param name: Operator name
        :param operator_namespace: Python module, from which operator was gotten.
        :param skip_trace: If it is set to True, the both operator and its internal calls
         to otherwise traced functions do not appear into the model graph.
        """
        self.name = name
        self.operator_namespace = operator_namespace
        self.skip_trace = skip_trace
