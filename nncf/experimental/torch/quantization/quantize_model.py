# Copyright (c) 2023 Intel Corporation
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

import torch

from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.data import Dataset
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.scopes import IgnoredScope
from nncf.torch.dynamic_graph.context import no_nncf_trace
from nncf.torch.dynamic_graph.io_handling import replicate_same_tensors
from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_inputs_with_objwalk
from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_outputs_with_objwalk
from nncf.torch.nested_objects_traversal import objwalk
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import get_model_device
from nncf.torch.utils import is_tensor
from nncf.torch.utils import training_mode_switcher


def create_nncf_network(model: torch.nn.Module, dataset: Dataset) -> NNCFNetwork:
    """
    Creates NNCFNetwork instance for the PyTorch model where the first item of dataset
    is used for model tracing.

    :param model: PyTorch model
    :param dataset: Dataset for model tracing
    :return: NNCFNetwork instance for the input model
    """

    def wrap_inputs(args):
        return wrap_nncf_model_inputs_with_objwalk(args, {})

    def wrap_outputs(retval):
        return wrap_nncf_model_outputs_with_objwalk(retval)

    def create_dummy_forward_fn(dataset, device):
        def dummy_forward(model):
            with no_nncf_trace():
                args = next(iter(dataset.get_inference_data()))

                def send_to_device(tensor):
                    return tensor.to(device)

                args = objwalk(args, is_tensor, send_to_device)

            args = wrap_inputs(args)
            retval = model(*args)
            retval = replicate_same_tensors(retval)
            return wrap_outputs(retval)

        return dummy_forward

    device = get_model_device(model)
    dummy_forward_fn = create_dummy_forward_fn(dataset, device)

    with training_mode_switcher(model, is_training=False):
        nncf_network = NNCFNetwork(
            model, dummy_forward_fn=dummy_forward_fn, wrap_inputs_fn=wrap_inputs, wrap_outputs_fn=wrap_outputs
        )

        nncf_network.nncf.get_tracing_context().disable_trace_dynamic_graph()

    return nncf_network


def quantize_impl(
    model: torch.nn.Module,
    calibration_dataset: Dataset,
    preset: QuantizationPreset,
    target_device: TargetDevice,
    subset_size: int,
    fast_bias_correction: bool,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
) -> torch.nn.Module:
    """
    Experimental implementation of the `quantize()` method for the PyTorch backend.
    """
    if fast_bias_correction is False:
        raise ValueError(f"fast_bias_correction={fast_bias_correction} is not supported")
    if target_device == TargetDevice.CPU_SPR:
        raise RuntimeError("target_device == CPU_SPR is not supported")

    if advanced_parameters is None:
        advanced_parameters = AdvancedQuantizationParameters()
    if not advanced_parameters.disable_bias_correction:
        nncf_logger.warning(
            "Bias correction and fast bias correction algorithms are not supported by Torch backend yet."
        )
        advanced_parameters.disable_bias_correction = True

    nncf_network = create_nncf_network(model.eval(), calibration_dataset)

    quantization_algorithm = PostTrainingQuantization(
        preset=preset,
        target_device=target_device,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        ignored_scope=ignored_scope,
        advanced_parameters=advanced_parameters,
    )

    quantized_model = quantization_algorithm.apply(nncf_network, dataset=calibration_dataset)

    # TODO (asuslov): quantized_model = quantized_model.strip()

    quantized_model.nncf.disable_dynamic_graph_building()

    return quantized_model
