"""
 Copyright (c) 2020 Intel Corporation
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
from typing import Callable, Any, Tuple, Dict

from torch.nn import Module
from torch.distributed import barrier

from nncf.torch.checkpoint_loading import load_state
from nncf.torch.composite_compression import PTCompositeCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.config import NNCFConfig
from nncf.torch.debug import set_debug_log_dir
from nncf.torch.dynamic_graph.graph_tracer import create_input_infos, create_dummy_forward_fn
from nncf.torch.graph.graph_builder import GraphBuilder
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import is_main_process
from nncf.torch.utils import is_dist_avail_and_initialized
from nncf.torch.algo_selector import COMPRESSION_ALGORITHMS

from nncf.common.utils.logger import logger


def get_compression_algorithm(config):
    algorithm_key = config.get('algorithm', 'NoCompressionAlgorithmBuilder')
    logger.info("Creating compression algorithm: {}".format(algorithm_key))
    return COMPRESSION_ALGORITHMS.get(algorithm_key)


def create_compressed_model(model: Module, config: NNCFConfig,
                            resuming_state_dict: dict = None,
                            dummy_forward_fn: Callable[[Module], Any] = None,
                            wrap_inputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
                            wrap_outputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
                            dump_graphs=True, ) \
    -> Tuple[PTCompressionAlgorithmController, NNCFNetwork]:
    """
    The main function used to produce a model ready for compression fine-tuning from an original PyTorch
    model and a configuration object.
    dummy_forward_fn
    :param model: The original model. Should have its parameters already loaded from a checkpoint or another
    source.
    :param config: A configuration object used to determine the exact compression modifications to be applied
    to the model
    :param resuming_state_dict: A PyTorch state dict object to load (strictly) into the compressed model after
    building.
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
    :param dump_graphs: Whether or not should also dump the internal graph representation of the
    original and compressed models in the .dot format into the log directory.
    :return: A controller for the compression algorithm (or algorithms, in which case the controller
    is an instance of CompositeCompressionController) and the model ready for compression parameter training wrapped
    as an object of NNCFNetwork."""

    if dummy_forward_fn is not None and wrap_inputs_fn is None:
        raise ValueError("A custom dummy forward function was specified, but the corresponding input wrapping function "
                         "was not. In case a custom dummy forward function is specified for purposes of NNCF graph "
                         "building, then the wrap_inputs_fn parameter MUST also be specified and be consistent with "
                         "the input wrapping done in dummy_forward_fn.")

    # Compress model that will be deployed for the inference on target device. No need to compress parts of the
    # model that are used on training stage only (e.g. AuxLogits of Inception-v3 model) or unused modules with weights.
    # As a consequence, no need to care about spoiling BN statistics, as there're disabled in eval mode.
    model.eval()

    if dump_graphs:
        if dummy_forward_fn is None:
            input_info_list = create_input_infos(config)
            graph_builder = GraphBuilder(custom_forward_fn=
                                         create_dummy_forward_fn(input_info_list,
                                                                 with_input_tracing=True))
        else:
            graph_builder = GraphBuilder(custom_forward_fn=dummy_forward_fn)

        if is_main_process():
            graph = graph_builder.build_graph(model)
            graph.visualize_graph(osp.join(config.get("log_dir", "."), "original_graph.dot"))

    set_debug_log_dir(config.get("log_dir", "."))

    input_info_list = create_input_infos(config)
    scopes_without_shape_matching = config.get('scopes_without_shape_matching', [])
    ignored_scopes = config.get('ignored_scopes')
    target_scopes = config.get('target_scopes')

    compressed_model = NNCFNetwork(model, input_infos=input_info_list,
                                   dummy_forward_fn=dummy_forward_fn,
                                   wrap_inputs_fn=wrap_inputs_fn,
                                   wrap_outputs_fn=wrap_outputs_fn,
                                   ignored_scopes=ignored_scopes,
                                   target_scopes=target_scopes,
                                   scopes_without_shape_matching=scopes_without_shape_matching)

    should_init = resuming_state_dict is None
    composite_builder = PTCompositeCompressionAlgorithmBuilder(config, should_init=should_init)
    composite_builder.apply_to(compressed_model)

    compression_ctrl = composite_builder.build_controller(compressed_model)

    # Required to ensure that the model leaving create_compressed_model has correct compressed graph.
    # In particular, this is currently required for correct functioning of RNNs.
    compressed_model.rebuild_graph()

    try:
        if resuming_state_dict is not None:
            load_state(compressed_model, resuming_state_dict, is_resume=True)
    finally:
        if dump_graphs and is_main_process():
            compressed_model_graph = compressed_model.get_graph()
            compressed_model_graph.visualize_graph(osp.join(config.get("log_dir", "."), "compressed_graph.dot"))

    # Synchronize all processes if run in distributed mode
    if is_dist_avail_and_initialized():
        try:
            barrier()
        # Exception can be raised during running barrier
        # if the backend not in the supported list https://pytorch.org/docs/stable/distributed.html
        except RuntimeError as err:
            logger.warning(err)
            logger.warning(
                "NNCF continues work, while does not guarantee that "
                "the processes will finish model's compression at the same time. "
                "If your training pipeline demands the processes be synchronized, please, "
                "keep attention to that error")
            return compression_ctrl, compressed_model

    return compression_ctrl, compressed_model
