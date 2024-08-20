# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.hardware.opset import HWConfigOpName
from nncf.torch.dynamic_graph.structs import NamespaceTarget

FX_OPERATOR_METATYPES = OperatorMetatypeRegistry("operator_metatypes")


@FX_OPERATOR_METATYPES.register()
class FXEmbeddingMetatype(OperatorMetatype):
    name = "EmbeddingOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["embedding"]}
    hw_config_names = [HWConfigOpName.EMBEDDING]
    weight_port_ids = [0]
