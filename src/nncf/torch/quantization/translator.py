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

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.quantization.quantizer_setup import QuantizationInsertionPointBase
from nncf.common.quantization.quantizer_setup import WeightQuantizationInsertionPoint
from nncf.torch.graph.transformations.commands import PTTargetPoint


class PTTargetPointTranslator:
    @staticmethod
    def translate(qip: QuantizationInsertionPointBase) -> PTTargetPoint:
        if isinstance(qip, WeightQuantizationInsertionPoint):
            return PTTargetPoint(target_type=TargetType.OPERATION_WITH_WEIGHTS, target_node_name=qip.target_node_name)
        assert isinstance(qip, ActivationQuantizationInsertionPoint)
        input_port_id = qip.input_port_id
        if input_port_id is not None:
            return PTTargetPoint(
                target_type=TargetType.OPERATOR_PRE_HOOK,
                target_node_name=qip.target_node_name,
                input_port_id=input_port_id,
            )
        return PTTargetPoint(target_type=TargetType.OPERATOR_POST_HOOK, target_node_name=qip.target_node_name)
