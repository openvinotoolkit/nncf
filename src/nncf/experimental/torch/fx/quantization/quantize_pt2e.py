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

from copy import deepcopy
from typing import Optional

import torch
import torch.fx
from torch.ao.quantization.pt2e.port_metadata_pass import PortNodeMetaForQDQ
from torch.ao.quantization.pt2e.utils import _disallow_eval_train
from torch.ao.quantization.pt2e.utils import _fuse_conv_bn_
from torch.ao.quantization.quantizer import Quantizer
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassManager

import nncf
from nncf import Dataset
from nncf import IgnoredScope
from nncf.common.factory import NNCFGraphFactory
from nncf.common.logging import nncf_logger
from nncf.common.utils.api_marker import api
from nncf.experimental.quantization.algorithms.post_training.algorithm import ExperimentalPostTrainingQuantization
from nncf.experimental.torch.fx.constant_folding import constant_fold
from nncf.experimental.torch.fx.quantization.quantizer.openvino_adapter import OpenVINOQuantizerAdapter
from nncf.experimental.torch.fx.quantization.quantizer.openvino_quantizer import OpenVINOQuantizer
from nncf.experimental.torch.fx.quantization.quantizer.torch_ao_adapter import TorchAOQuantizerAdapter
from nncf.experimental.torch.fx.transformations import QUANTIZE_NODE_TARGETS
from nncf.experimental.torch.fx.transformations import DuplicateDQPassNoAnnotations
from nncf.experimental.torch.fx.transformations import compress_post_quantize_transformation
from nncf.quantization.advanced_parameters import AdvancedBiasCorrectionParameters
from nncf.quantization.advanced_parameters import AdvancedSmoothQuantParameters
from nncf.quantization.range_estimator import RangeEstimatorParameters


@api(canonical_alias="nncf.experimental.torch.fx.quantize_pt2e")
def quantize_pt2e(
    model: torch.fx.GraphModule,
    quantizer: Quantizer,
    calibration_dataset: Dataset,
    subset_size: int = 300,
    fast_bias_correction: Optional[bool] = True,
    smooth_quant: bool = False,
    bias_correction_params: Optional[AdvancedBiasCorrectionParameters] = None,
    smooth_quant_params: Optional[AdvancedSmoothQuantParameters] = None,
    activations_range_estimator_params: Optional[RangeEstimatorParameters] = None,
    weights_range_estimator_params: Optional[RangeEstimatorParameters] = None,
    batchwise_statistics: Optional[bool] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    fold_quantize: bool = True,
    do_copy: bool = False,
) -> torch.fx.GraphModule:
    """
    Applies post-training quantization to the torch.fx.GraphModule provided model
    using provided torch.ao quantizer.

    :param model: A torch.fx.GraphModule instance to be quantized.
    :param quantizer: Torch ao quantizer to annotate nodes in the graph with quantization setups
        to convey the desired way of quantization.
    :param calibration_dataset: A representative dataset for the
        calibration process.
    :param subset_size: Size of a subset to calculate activations
        statistics used for quantization.
    :param fast_bias_correction: Setting this option to `False` enables a different
        bias correction method which is more accurate, in general, and takes
        more time but requires less memory. None disables the bias correction algorithm.
    :param smooth_quant: Setting this option to `True` enables the SmoothQuant algorithm.
    :param bias_correction_params: Contains advanced parameters for fine-tuning bias correction algorithm.
    :param smooth_quant_params: Contains advanced alpha parameters for SmoothQuant algorithm.
    :param activations_range_estimator_params: Contains parameters for estimating the range
        of activations of the model.
    :param weights_range_estimator_params: Contains parameters for estimating the range
        of weights of the model.
    :param batchwise_statistics: Determines whether quantizer statistics should be calculated
        for each item of the batch or for the entire batch, default is None, which means
        it set True if batch_size > 1 otherwise False.
    :param ignored_scope: An ignored scope that defined the list of model control
        flow graph nodes to be ignored during quantization.
    :param fold_quantize: Boolean flag for whether fold the quantize op or not. The value is True by default.
    :param do_copy: The copy of the given model is being quantized if do_copy == True,
        otherwise the model is quantized inplace. Default value is False.
    :return: The quantized torch.fx.GraphModule instance.
    """
    nncf_logger.warning("This is an experimental feature and may change in the future without notice.")

    if subset_size < 1:
        msg = "Subset size must be positive."
        raise nncf.ValidationError(msg)

    batch_size = calibration_dataset.get_batch_size()
    if batchwise_statistics is None:
        batchwise_statistics = batch_size is not None and batch_size > 1

    original_graph_meta = model.meta

    if do_copy:
        model = deepcopy(model)

    _fuse_conv_bn_(model)
    if isinstance(quantizer, OpenVINOQuantizer) or hasattr(quantizer, "get_nncf_quantization_setup"):
        quantizer = OpenVINOQuantizerAdapter(quantizer)
    else:
        quantizer = TorchAOQuantizerAdapter(quantizer)

    # Call transform_prior_quantization before the NNCFGraph creation
    transformed_model = quantizer.transform_prior_quantization(model)

    quantization_algorithm = ExperimentalPostTrainingQuantization(
        quantizer=quantizer,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        smooth_quant=smooth_quant,
        bias_correction_params=bias_correction_params,
        smooth_quant_params=smooth_quant_params,
        activations_range_estimator_params=activations_range_estimator_params,
        weights_range_estimator_params=weights_range_estimator_params,
        ignored_scope=ignored_scope,
        batchwise_statistics=batchwise_statistics,
    )

    nncf_graph = NNCFGraphFactory.create(transformed_model)
    quantized_model = quantization_algorithm.apply(transformed_model, nncf_graph, dataset=calibration_dataset)

    # Magic. Without this call compiled model is not performant
    quantized_model = GraphModule(quantized_model, quantized_model.graph)

    if fold_quantize:
        if isinstance(quantizer, OpenVINOQuantizerAdapter):
            compress_post_quantize_transformation(quantized_model)
        else:
            constant_fold(quantized_model, _quant_node_constraint)

    pm = PassManager([DuplicateDQPassNoAnnotations()])

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
    """
    If there is any pure ops between get_attr and quantize op they will be const propagated
    e.g. get_attr(weight) -> transpose -> quantize -> dequantize*
    (Note: dequantize op is not going to be constant propagated)

    This filter is added because we don't want to constant fold the things that are not
    related to quantization
    """
    return n.op == "call_function" and n.target in QUANTIZE_NODE_TARGETS
