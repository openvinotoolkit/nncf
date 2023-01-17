"""
 Copyright (c) 2020-2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from os import path as osp
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch.distributed import barrier
from torch.nn import Module

from nncf.api.compression import CompressionAlgorithmController
from nncf.common.compression import BaseCompressionAlgorithmController as BaseController
from nncf.common.utils.debug import set_debug_log_dir
from nncf.common.logging import nncf_logger
from nncf.config import NNCFConfig
from nncf.config.extractors import extract_algorithm_names
from nncf.config.structures import ModelEvaluationArgs
from nncf.config.telemetry_extractors import CompressionStartedFromConfig
from nncf.config.utils import is_accuracy_aware_training
from nncf.telemetry import tracked_function
from nncf.telemetry.events import NNCF_PT_CATEGORY
from nncf.torch.algo_selector import NoCompressionAlgorithmBuilder
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.torch.composite_compression import PTCompositeCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.dynamic_graph.graph_tracer import create_input_infos
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import is_dist_avail_and_initialized
from nncf.torch.utils import is_main_process
# pylint:disable=too-many-branches
from nncf.torch.utils import maybe_convert_legacy_names_in_compress_state


@tracked_function(NNCF_PT_CATEGORY, [CompressionStartedFromConfig(argname="config"), ])
def create_compressed_model(model: Module,
                            config: NNCFConfig,
                            compression_state: Optional[Dict[str, Any]] = None,
                            dummy_forward_fn: Callable[[Module], Any] = None,
                            wrap_inputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
                            wrap_outputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
                            dump_graphs=True) \
    -> Tuple[CompressionAlgorithmController, NNCFNetwork]:
    """
    The main function used to produce a model ready for compression fine-tuning from an original PyTorch
    model and a configuration object.
    dummy_forward_fn
    :param model: The original model. Should have its parameters already loaded from a checkpoint or another
    source.
    :param config: A configuration object used to determine the exact compression modifications to be applied
    to the model
    :param compression_state: representation of the entire compression state to unambiguously restore
    the compressed model. Includes builder and controller states.
    :param dummy_forward_fn: if supplied, will be used instead of a *forward* function call to build
    the internal graph representation via tracing. Specifying this is useful when the original training pipeline
    has special formats of data loader output or has additional *forward* arguments other than input tensors.
    Otherwise, the *forward* call of the model during graph tracing will be made with mock tensors according
    to the shape specified in the config object. The dummy_forward_fn code MUST contain calls to nncf.nncf_model_input
    functions made with each compressed model input tensor in the underlying model's args/kwargs tuple, and these
    calls should be exactly the same as in the wrap_inputs_fn function code (see below); if dummy_forward_fn is
    specified, then wrap_inputs_fn also must be specified.
    :param wrap_inputs_fn: if supplied, will be used on the module's input arguments during a regular, non-dummy
    forward call before passing the inputs to the underlying compressed model. This is required if the model's input
    tensors that are important for compression are not supplied as arguments to the model's forward call directly, but
    instead are located in a container (such as list), and the model receives the container as an argument.
    wrap_inputs_fn should take as input two arguments - the tuple of positional arguments to the underlying
    model's forward call, and a dict of keyword arguments to the same. The function should wrap each tensor among the
    supplied model's args and kwargs that is important for compression (e.g. quantization) with an nncf.nncf_model_input
    function, which is a no-operation function and marks the tensors as inputs to be traced by NNCF in the internal
    graph representation. Output is the tuple of (args, kwargs), where args and kwargs are the same as were supplied in
    input, but each tensor in the original input. Must be specified if dummy_forward_fn is specified.
    :param wrap_outputs_fn: same as `wrap_inputs_fn`, but applies to model outputs
    :param dump_graphs: Whether to dump the internal graph representation of the
    original and compressed models in the .dot format into the log directory.
    :return: A controller for the compression algorithm (or algorithms, in which case the controller
    is an instance of CompositeCompressionController) and the model ready for compression parameter training wrapped
    as an object of NNCFNetwork."""

    set_debug_log_dir(config.get("log_dir", "."))

    is_legacy_model_state_dict = compression_state is not None and \
                                 BaseController.BUILDER_STATE not in compression_state and \
                                 BaseController.CONTROLLER_STATE not in compression_state
    maybe_convert_legacy_names_in_compress_state(compression_state)

    should_init = compression_state is None

    nncf_network = create_nncf_network(model, config, dummy_forward_fn, wrap_inputs_fn, wrap_outputs_fn)

    if dump_graphs and is_main_process():
        nncf_network.get_graph().visualize_graph(osp.join(config.get("log_dir", "."), "original_graph.dot"))

    builder = create_compression_algorithm_builder(config, should_init)

    is_state_loadable = not is_legacy_model_state_dict and compression_state is not None
    if is_state_loadable:
        builder.load_state(compression_state[BaseController.BUILDER_STATE])
    compressed_model = builder.apply_to(nncf_network)
    compression_ctrl = builder.build_controller(compressed_model)
    if is_state_loadable:
        compression_ctrl.load_state(compression_state[BaseController.CONTROLLER_STATE])

    # Required to ensure that the model leaving create_compressed_model has correct compressed graph.
    # In particular, this is currently required for correct functioning of RNNs.
    compressed_model.rebuild_graph()

    try:
        if is_legacy_model_state_dict:
            from nncf.torch import load_state  # pylint: disable=cyclic-import
            state_dict_to_load = compression_state.get('state_dict', compression_state)
            load_state(compressed_model, state_dict_to_load, is_resume=True)
    finally:
        if dump_graphs and is_main_process():
            compressed_model_graph = compressed_model.get_graph()
            compressed_model_graph.visualize_graph(osp.join(config.get("log_dir", "."), "compressed_graph.dot"))

    synchronize_all_processes_in_distributed_mode()
    return compression_ctrl, compressed_model


def create_nncf_network(model,
                        config: NNCFConfig,
                        dummy_forward_fn: Callable[[Module], Any] = None,
                        wrap_inputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
                        wrap_outputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None) -> NNCFNetwork:
    """
    The main function used to produce a model ready for adding compression from an original PyTorch
    model and a configuration object.
    dummy_forward_fn
    :param model: The original model. Should have its parameters already loaded from a checkpoint or another
    source.
    :param config: A configuration object used to determine the exact compression modifications to be applied
    to the model
    :param dummy_forward_fn: if supplied, will be used instead of a *forward* function call to build
    the internal graph representation via tracing. Specifying this is useful when the original training pipeline
    has special formats of data loader output or has additional *forward* arguments other than input tensors.
    Otherwise, the *forward* call of the model during graph tracing will be made with mock tensors according
    to the shape specified in the config object. The dummy_forward_fn code MUST contain calls to nncf.nncf_model_input
    functions made with each compressed model input tensor in the underlying model's args/kwargs tuple, and these
    calls should be exactly the same as in the wrap_inputs_fn function code (see below); if dummy_forward_fn is
    specified, then wrap_inputs_fn also must be specified.
    :param wrap_inputs_fn: if supplied, will be used on the module's input arguments during a regular, non-dummy
    forward call before passing the inputs to the underlying compressed model. This is required if the model's input
    tensors that are important for compression are not supplied as arguments to the model's forward call directly, but
    instead are located in a container (such as list), and the model receives the container as an argument.
    wrap_inputs_fn should take as input two arguments - the tuple of positional arguments to the underlying
    model's forward call, and a dict of keyword arguments to the same. The function should wrap each tensor among the
    supplied model's args and kwargs that is important for compression (e.g. quantization) with an nncf.nncf_model_input
    function, which is a no-operation function and marks the tensors as inputs to be traced by NNCF in the internal
    graph representation. Output is the tuple of (args, kwargs), where args and kwargs are the same as were supplied in
    input, but each tensor in the original input. Must be specified if dummy_forward_fn is specified.
    :param wrap_outputs_fn: if supplied, will be used on the module's output during a regular, non-dummy forward call.
    :return: A model wrapped by NNCFNetwork, which is ready for adding compression. """

    if dummy_forward_fn is not None and wrap_inputs_fn is None:
        raise ValueError(
            "A custom dummy forward function was specified, but the corresponding input wrapping function "
            "was not. In case a custom dummy forward function is specified for purposes of NNCF graph "
            "building, then the wrap_inputs_fn parameter MUST also be specified and be consistent with "
            "the input wrapping done in dummy_forward_fn.")

    # Compress model that will be deployed for the inference on target device. No need to compress parts of the
    # model that are used on training stage only (e.g. AuxLogits of Inception-v3 model) or unused modules with weights.
    # As a consequence, no need to care about spoiling BN statistics, as they're disabled in eval mode.
    model.eval()

    input_info_list = create_input_infos(config)
    scopes_without_shape_matching = config.get('scopes_without_shape_matching', [])
    ignored_scopes = config.get('ignored_scopes')
    target_scopes = config.get('target_scopes')

    original_model_accuracy = None
    if is_accuracy_aware_training(config):
        if config.has_extra_struct(ModelEvaluationArgs):
            evaluation_args = config.get_extra_struct(ModelEvaluationArgs)
            with torch.no_grad():
                original_model_accuracy = evaluation_args.eval_fn(model)
                nncf_logger.info(f"Uncompressed model accuracy = {original_model_accuracy}")

    nncf_network = NNCFNetwork(model, input_infos=input_info_list,
                               dummy_forward_fn=dummy_forward_fn,
                               wrap_inputs_fn=wrap_inputs_fn,
                               wrap_outputs_fn=wrap_outputs_fn,
                               ignored_scopes=ignored_scopes,
                               target_scopes=target_scopes,
                               scopes_without_shape_matching=scopes_without_shape_matching,
                               original_model_accuracy=original_model_accuracy)

    nncf_network.get_tracing_context().disable_trace_dynamic_graph()

    synchronize_all_processes_in_distributed_mode()
    return nncf_network


def synchronize_all_processes_in_distributed_mode():
    if is_dist_avail_and_initialized():
        try:
            barrier()
        # Exception can be raised during running barrier
        # if the backend not in the supported list https://pytorch.org/docs/stable/distributed.html
        except RuntimeError as err:
            nncf_logger.warning("Training pipeline spawned an error while "
                                "synchronizing distributed training processes:")
            nncf_logger.warning(err)
            nncf_logger.warning("Desynchronization of distributed processes may occur.")


def create_compression_algorithm_builder(config: NNCFConfig, should_init=True) -> PTCompressionAlgorithmBuilder:
    """
    Create compression algorithm builders by a given list of algorithm names.

    :param config: A configuration object used to determine the exact compression modifications to be applied
    to the model
    :param should_init: The flag indicates that the generated compression builder will initialize (True) or not (False)
    the training parameters of the model during model building.
    :return: compression algorithm builder
    """
    algo_names = extract_algorithm_names(config)
    return create_compression_algorithm_builder_from_algo_names(algo_names, config, should_init)


def create_compression_algorithm_builder_from_algo_names(algo_names: List[str],
                                                         config: NNCFConfig,
                                                         should_init: bool) -> PTCompressionAlgorithmBuilder:
    """
    Create compression algorithm builders by a given list of algorithm names.

    :param algo_names: list of algorithm names
    :param config: A configuration object used to determine the exact compression modifications to be applied
    to the model
    :param should_init: The flag indicates that the generated compression builder will initialize (True) or not (False)
    the training parameters of the model during model building.
    :return: compression algorithm builder
    """
    if not algo_names:
        algo_builder_classes = [NoCompressionAlgorithmBuilder]
    else:
        algo_builder_classes = [PT_COMPRESSION_ALGORITHMS.get(algo_name) for algo_name in algo_names]
    if len(algo_builder_classes) == 1:
        builder = next(iter(algo_builder_classes))(config, should_init=should_init)
    else:
        builder = PTCompositeCompressionAlgorithmBuilder(config, should_init=should_init)
    return builder
