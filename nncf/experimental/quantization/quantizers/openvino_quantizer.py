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

from collections import defaultdict
from typing import Dict, List, Optional, Union

import torch.fx
from torch.ao.quantization.observer import HistogramObserver
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.ao.quantization.quantizer.quantizer import QuantizationAnnotation as TorchAOQuantizationAnnotation
from torch.ao.quantization.quantizer.quantizer import QuantizationSpec as TorchAOQuantizationSpec
from torch.ao.quantization.quantizer.quantizer import Quantizer as TorchAOQuantizer

from nncf.common.graph.graph import NNCFGraph
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationRule
from nncf.common.quantization.quantizer_setup import QuantizationPointBase
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizationScheme
from nncf.common.quantization.structs import QuantizerConfig as NNCFQuantizerConfig
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter
from nncf.experimental.torch.fx.node_utils import get_graph_node_by_name
from nncf.experimental.torch.fx.transformations import fold_constant_except_qdq
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import FP8QuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.advanced_parameters import QuantizationParameters
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.scopes import IgnoredScope

QUANT_ANNOTATION_KEY = "quantization_annotation"


class OpenVINOQuantizer(TorchAOQuantizer):
    """
    Implementation of the Torch AO quantizer which annotates models with quantization annotations
    optimally for the inference via OpenVINO.
    """

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
        quantizer_propagation_rule: QuantizerPropagationRule = QuantizerPropagationRule.MERGE_ALL_IN_ONE,
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
        MERGE_ALL_IN_ONE by default.
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
        self._min_max_algo._set_backend_entity(model)
        return self._min_max_algo.find_quantization_setup(model, nncf_graph)

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        nncf_grpah = GraphConverter.create_nncf_graph(model)
        quantization_setup = self.get_quantization_setup(model, nncf_grpah)
        target_node_vs_qp: Dict[str, List[QuantizationPointBase]] = defaultdict(list)
        graph = model.graph
        for qp in quantization_setup.quantization_points.values():
            target_node_vs_qp[qp.insertion_point.target_node_name].append(qp)

        for target_node_name, qps in target_node_vs_qp.items():
            input_qspec_map = dict()
            output_qspec = None
            target_node = get_graph_node_by_name(graph, target_node_name)
            for qp in qps:
                ip = qp.insertion_point
                if qp.is_activation_quantization_point():
                    inductor_qspec = self._convert_nncf_qspec_to_inductor_qspec(qp.qconfig, is_weight=False)
                    if ip.input_port_id is None:
                        output_qspec = inductor_qspec
                    else:
                        node = target_node.all_input_nodes[ip.input_port_id]
                        input_qspec_map[node] = inductor_qspec
                else:
                    inductor_qspec = self._convert_nncf_qspec_to_inductor_qspec(qp.qconfig, is_weight=True)
                    weight_node = target_node.all_input_nodes[1]
                    input_qspec_map[weight_node] = inductor_qspec

            annotation = TorchAOQuantizationAnnotation(input_qspec_map=input_qspec_map, output_qspec=output_qspec)
            assert QUANT_ANNOTATION_KEY not in target_node.meta
            target_node.meta[QUANT_ANNOTATION_KEY] = annotation

    def _convert_nncf_qspec_to_inductor_qspec(
        self, qspec: NNCFQuantizerConfig, is_weight: bool
    ) -> TorchAOQuantizationSpec:
        # Eps value is borrowed from
        # https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/quantizer/x86_inductor_quantizer.py
        # get_default_x86_inductor_quantization_config
        extra_args = {"eps": 2**-12}
        if qspec.per_channel:
            torch_qscheme = (
                torch.per_channel_symmetric if qspec.mode is QuantizationScheme.SYMMETRIC else torch.per_channel_affine
            )
        else:
            torch_qscheme = (
                torch.per_tensor_symmetric if qspec.mode is QuantizationScheme.SYMMETRIC else torch.per_tensor_affine
            )
        if is_weight:
            observer = PerChannelMinMaxObserver
            quant_min = -128
            quant_max = 127
            dtype = torch.int8
            channel_axis = 0
        else:
            observer = (
                HistogramObserver
                if torch_qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]
                else PerChannelMinMaxObserver
            )
            quant_min = 0
            quant_max = 255
            dtype = torch.int8 if qspec.signedness_to_force else torch.uint8
            channel_axis = 1  # channel dim for activations
        return TorchAOQuantizationSpec(
            dtype=dtype,
            observer_or_fake_quant_ctr=observer.with_args(**extra_args),
            quant_min=quant_min,
            quant_max=quant_max,
            qscheme=torch_qscheme,
            ch_axis=channel_axis,
            is_dynamic=False,
        )

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    def transform_for_annotation(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        fold_constant_except_qdq(model)
        return model
