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

from copy import deepcopy
from typing import Optional

import torch
import torch.fx
from torch.ao.quantization.pt2e.duplicate_dq_pass import DuplicateDQPass
from torch.ao.quantization.pt2e.port_metadata_pass import PortNodeMetaForQDQ
from torch.ao.quantization.pt2e.qat_utils import _fold_conv_bn_qat
from torch.ao.quantization.pt2e.utils import _disallow_eval_train
from torch.ao.quantization.quantizer import Quantizer
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassManager

from nncf.common.factory import NNCFGraphFactory
from nncf.common.logging import nncf_logger
from nncf.data import Dataset
from nncf.experimental.common.quantization.algorithms.post_training.algorithm import (
    ExperimentalPostTrainingQuantization,
)
from nncf.experimental.common.quantization.algorithms.quantizer.base_quantizer import NNCFQuantizer
from nncf.experimental.common.quantization.algorithms.quantizer.fx_quantizer import NNCFFXQuantizer
from nncf.experimental.torch.fx.constant_folding import constant_fold
from nncf.experimental.torch.fx.transformations import QUANTIZE_NODE_TARGETS
from nncf.experimental.torch.fx.transformations import fuse_conv_bn
from nncf.quantization.advanced_parameters import AdvancedBiasCorrectionParameters
from nncf.quantization.advanced_parameters import AdvancedSmoothQuantParameters
from nncf.quantization.advanced_parameters import RangeEstimatorParameters


def quantize_pt2e(
    model: torch.fx.GraphModule,
    quantizer: Quantizer,
    calibration_dataset: Dataset,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    smooth_quant: bool = False,
    bias_correction_params: Optional[AdvancedBiasCorrectionParameters] = None,
    smooth_quant_params: Optional[AdvancedSmoothQuantParameters] = None,
    activations_range_estimator_params: Optional[RangeEstimatorParameters] = None,
    weights_range_estimator_params: Optional[RangeEstimatorParameters] = None,
    fold_quantize: Optional[bool] = False,
) -> torch.fx.GraphModule:
    """
    Implementation of the `quantize()` method for the Torch FX backend.
    """
    nncf_logger.warning(
        "Experimental Torch FX quantization backend is being used for the given torch.fx.GraphModule model."
        " Torch FX PTQ is an experimental feature, consider using Torch or OpenVino PTQ backends"
        " in case of errors or a poor model performance."
    )

    original_graph_meta = model.meta

    copied_model = deepcopy(model)

    # To make it easier for bias correction algorithms,
    # biases are being separated by the followng calls.
    fuse_conv_bn(copied_model)
    # Call ao quantizer transform_for_annotation
    # before the NNCFGraph creation
    quantizer.transform_for_annotation(copied_model)

    if not isinstance(quantizer, NNCFQuantizer):
        quantizer = NNCFFXQuantizer(quantizer)

    quantization_algorithm = ExperimentalPostTrainingQuantization(
        quantizer=quantizer,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        smooth_quant=smooth_quant,
        bias_correction_params=bias_correction_params,
        smooth_quant_params=smooth_quant_params,
        activations_range_estimator_params=activations_range_estimator_params,
        weights_range_estimator_params=weights_range_estimator_params,
    )

    nncf_graph = NNCFGraphFactory.create(copied_model)
    quantized_model = quantization_algorithm.apply(copied_model, nncf_graph, dataset=calibration_dataset)

    # Magic. Without this call compiled model
    # is not preformant
    quantized_model = GraphModule(quantized_model, quantized_model.graph)

    quantized_model = _fold_conv_bn_qat(quantized_model)
    if fold_quantize:
        constant_fold(quantized_model, _quant_node_constraint)

    pm = PassManager([DuplicateDQPass()])

    quantized_model = pm(quantized_model).graph_module
    pm = PassManager([PortNodeMetaForQDQ()])
    quantized_model = pm(quantized_model).graph_module

    quantized_model.meta.update(original_graph_meta)
    quantized_model = _disallow_eval_train(quantized_model)
    # Each transformation adds a duplicate tensor value to the model buffer.
    #  This step removes the duplicates tensor values from the buffer.
    quantized_model = GraphModule(quantized_model, quantized_model.graph)

    return quantized_model


def _quant_node_constraint(n: torch.fx.Node) -> bool:
    """If there is any pure ops between get_attr and quantize op they will be const propagated
    e.g. get_attr(weight) -> transpose -> quantize -> dequantize*
    (Note: dequantize op is not going to be constant propagated)

    This filter is added because we don't want to constant fold the things that are not
    related to quantization
    """
    return n.op == "call_function" and n.target in QUANTIZE_NODE_TARGETS
