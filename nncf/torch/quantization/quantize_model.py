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

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import torch

from nncf.common.quantization.structs import QuantizationPreset
from nncf.data import Dataset
from nncf.parameters import CompressWeightsMode
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
from nncf.torch.nncf_module_replacement import replace_modules_by_nncf_modules
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.weights_compression import insert_pre_compression_operations
from nncf.torch.utils import get_model_device
from nncf.torch.utils import is_tensor
from nncf.torch.utils import training_mode_switcher


def create_nncf_network(model: torch.nn.Module, dataset: Dataset) -> NNCFNetwork:
    """
    Creates NNCFNetwork instance for the PyTorch model where the first item of dataset
    is used for model tracing.

    :param model: PyTorch model.
    :param dataset: Dataset for model tracing.
    :return: NNCFNetwork instance for the input model.
    """

    def get_inputs(dataloader_output: Any) -> Tuple[Tuple, Dict]:
        if isinstance(dataloader_output, dict):
            return (), dataloader_output
        if isinstance(dataloader_output, tuple):
            return dataloader_output, {}
        return (dataloader_output,), {}

    def wrap_inputs(args, kwargs):
        return wrap_nncf_model_inputs_with_objwalk(args, kwargs)

    def wrap_outputs(retval):
        return wrap_nncf_model_outputs_with_objwalk(retval)

    def create_dummy_forward_fn(dataset, device):
        def dummy_forward(model):
            with no_nncf_trace():
                data = next(iter(dataset.get_inference_data()))
                args, kwargs = get_inputs(data)

                def send_to_device(tensor):
                    return tensor.to(device)

                args = objwalk(args, is_tensor, send_to_device)
                kwargs = objwalk(kwargs, is_tensor, send_to_device)

            args, kwargs = wrap_inputs(args, kwargs)
            retval = model(*args, **kwargs)
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
    preset: Union[QuantizationPreset, None],
    target_device: TargetDevice,
    subset_size: int,
    fast_bias_correction: bool,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
) -> torch.nn.Module:
    """
    Implementation of the `quantize()` method for the PyTorch backend.
    """
    if fast_bias_correction is False:
        raise ValueError(f"fast_bias_correction={fast_bias_correction} is not supported")
    if target_device == TargetDevice.CPU_SPR:
        raise RuntimeError("target_device == CPU_SPR is not supported")

    copied_model = deepcopy(model)
    nncf_network = create_nncf_network(copied_model.eval(), calibration_dataset)

    quantization_algorithm = PostTrainingQuantization(
        preset=preset,
        target_device=target_device,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        ignored_scope=ignored_scope,
        advanced_parameters=advanced_parameters,
    )

    quantized_model = quantization_algorithm.apply(
        nncf_network, nncf_network.nncf.get_graph(), dataset=calibration_dataset
    )

    quantized_model.nncf.disable_dynamic_graph_building()

    return quantized_model


def compress_weights_impl(
    model: torch.nn.Module,
    mode=CompressWeightsMode.INT8,
    ratio: Optional[float] = None,
    group_size: Optional[int] = None,
    ignored_scope: Optional[IgnoredScope] = None,
) -> torch.nn.Module:
    """
    Implementation of the `compress_weights()` method for the PyTorch backend. Currently it supports INT8
    mode only with default ratio and group_size.

    :param model: a Torch model for compression.
    :param mode: Defines a mode for weight compression.
        INT8 stands for 8-bit integer quantization of all weights.
        INT4_SYM stands for a mixed-precision weights quantization with 4-bit integer as a primary precision.
            Weights are quantized to a primary precision symmetrically with a fixed zero point equals to 8.
            The first and the last layers are always compressed to a backup precision, which is 8-bit integer,
            by default. All others are quantized whether to 4-bit integer or to a backup precision depending on
            criteria and the given ratio.
        INT4_ASYM is the same as INT4_SYM mode, but weights are quantized to a primary precision asymmetrically
            with a typical non-fixed zero point.
        NF4 is the same as INT4_SYM mode, but primary precision is NF4 data type without zero point.
    :param ratio: the ratio between baseline and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
        and the rest to INT8).
    :param group_size: number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping.
    :param ignored_scope: An ignored scope that defined the list of model control
        flow graph nodes to be ignored during quantization.
    :return: The non-trainable model with compressed weights and dequantization operations.
    """
    if ignored_scope is not None:
        raise AttributeError("Torch backend does not support ignored scope.")
    if mode != CompressWeightsMode.INT8:
        raise AttributeError(f"Torch backend supports only INT8 mode for weight compression, but given {mode} mode.")
    compressed_model, _ = replace_modules_by_nncf_modules(model)
    insert_pre_compression_operations(model)

    return compressed_model
