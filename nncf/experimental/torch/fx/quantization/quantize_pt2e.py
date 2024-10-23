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
from nncf.experimental.common.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.experimental.common.quantization.algorithms.quantizer.fx_quantizer import NNCFFXQuantizer
from nncf.experimental.torch.fx.transformations import fuse_conv_bn
from nncf.parameters import ModelType
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters

DEFAULT_RANGE_TYPE = "mean_min_max"


def quantize_pt2e(
    model: torch.fx.GraphModule,
    quantizer: Quantizer,
    calibration_dataset: Dataset,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[ModelType] = None,
    advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
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

    quantization_algorithm = PostTrainingQuantization(
        quantizer=NNCFFXQuantizer(quantizer),
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        advanced_parameters=advanced_parameters,
    )

    # To make it easier for bias correction algorithms,
    # biases are being separated by the followng calls.
    fuse_conv_bn(copied_model)

    nncf_graph = NNCFGraphFactory.create(copied_model)
    quantized_model = quantization_algorithm.apply(copied_model, nncf_graph, dataset=calibration_dataset)

    # Magic. Without this call compiled model
    # is not preformant
    quantized_model = GraphModule(quantized_model, quantized_model.graph)

    quantized_model = _fold_conv_bn_qat(quantized_model)
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
