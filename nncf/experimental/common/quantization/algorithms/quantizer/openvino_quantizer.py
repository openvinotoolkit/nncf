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

from typing import Optional, Union

import torch.fx

from nncf.common.graph.graph import NNCFGraph
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationRule
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.structs import QuantizationPreset
from nncf.experimental.common.quantization.algorithms.quantizer.base_quantizer import NNCFQuantizer
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import FP8QuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.advanced_parameters import QuantizationParameters
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.scopes import IgnoredScope


class OpenVINOQuantizer(NNCFQuantizer):
    def __init__(
        self,
        mode: Optional[QuantizationMode] = None,
        preset: Optional[QuantizationPreset] = None,
        target_device: TargetDevice = TargetDevice.ANY,
        model_type: Optional[ModelType] = None,
        ignored_scope: Optional[IgnoredScope] = None,
        overflow_fix: Optional[OverflowFix] = None,
        quantize_outputs: bool = False,
        activations_quantization_params: Union[QuantizationParameters, FP8QuantizationParameters] = None,
        weights_quantization_params: Union[QuantizationParameters, FP8QuantizationParameters] = None,
        quantizer_propagation_rule: Optional[QuantizerPropagationRule] = None,
    ):
        """
        :param mode: Defines optimization mode for the algorithm. None by default.
        :param preset: A preset controls the quantization mode (symmetric and asymmetric).
            It can take the following values:
            - `performance`: Symmetric quantization of weights and activations.
            - `mixed`: Symmetric quantization of weights and asymmetric quantization of activations.
            Default value is None. In this case, `mixed` preset is used for `transformer`
            model type otherwise `performance`.
        :param target_device: A target device the specificity of which will be taken
            into account while compressing in order to obtain the best performance
            for this type of device, defaults to TargetDevice.ANY.
        :param model_type: Model type is needed to specify additional patterns
            in the model. Supported only `transformer` now.
        :param ignored_scope: An ignored scope that defined the list of model control
            flow graph nodes to be ignored during quantization.
        :param overflow_fix: This option controls whether to apply the overflow issue
            fix for the 8-bit quantization.
        :param quantize_outputs: Whether to insert additional quantizers right before
            each of the model outputs.
        :param activations_quantization_params: Quantization parameters for model
            activations.
        :param weights_quantization_params: Quantization parameters for model weights.
        :param quantizer_propagation_rule: The strategy to be used while propagating and merging quantizers.
        """
        self._min_max_algo = MinMaxQuantization(
            mode=mode,
            preset=preset,
            target_device=target_device,
            model_type=model_type,
            ignored_scope=ignored_scope,
            overflow_fix=overflow_fix,
            quantize_outputs=quantize_outputs,
            activations_quantization_params=activations_quantization_params,
            weights_quantization_params=weights_quantization_params,
            quantizer_propagation_rule=quantizer_propagation_rule,
        )

    def get_quantization_setup(self, model: torch.fx.GraphModule, nncf_graph: NNCFGraph) -> SingleConfigQuantizerSetup:
        """
        Builds SingleConfigQuantizerSetup for the given model.

        :param model: Backend-specific model, for which Quantization Target Points are being seek.
        :param nncf_graph: NNCFGraph instance.
        :return: SingleConfigQuantizerSetup for the given model.
        """
        self._min_max_algo._set_backend_entity(model)
        return self._min_max_algo._find_quantization_setup(model, nncf_graph)
