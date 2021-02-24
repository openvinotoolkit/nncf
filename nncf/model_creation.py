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

from nncf.checkpoint_loading import load_state
from nncf.composite_compression import PTCompositeCompressionAlgorithmBuilder
from nncf.compression_method_api import PTCompressionAlgorithmController
from nncf.config import NNCFConfig
from nncf.debug import set_debug_log_dir
from nncf.dynamic_graph.graph_builder import GraphBuilder, create_input_infos, create_dummy_forward_fn
from nncf.nncf_network import NNCFNetwork
from nncf.utils import is_main_process
from nncf.algo_selector import COMPRESSION_ALGORITHMS

from nncf.common.utils.logger import logger


def get_compression_algorithm(config):
    algorithm_key = config.get('algorithm', 'NoCompressionAlgorithmBuilder')
    logger.info("Creating compression algorithm: {}".format(algorithm_key))
    return COMPRESSION_ALGORITHMS.get(algorithm_key)


def create_compressed_model(model: Module, config: NNCFConfig,
                            resuming_state_dict: dict = None,
                            dummy_forward_fn: Callable[[Module], Any] = None,
                            wrap_inputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
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
    to the shape specified in the config object.
    :param wrap_inputs_fn: if supplied, will be used on the module's input arguments during a regular, non-dummy
    forward call before passing the inputs to the underlying compressed model. This is required if the model's input
    tensors that are important for compression are not supplied as arguments to the model's forward call directly, but
    instead are located in a container (such as list), and the model receives the container as an argument.
    wrap_inputs_fn should take as input two arguments - the tuple of positional arguments to the underlying
    model's forward call, and a dict of keyword arguments to the same. The function should wrap each tensor among the
    supplied model's args and kwargs that is important for compression (e.g. quantization) with an nncf.nncf_model_input
    function, which is a no-operation function and marks the tensors as inputs to be traced by NNCF in the internal
    graph representation. Output is the tuple of (args, kwargs), where args and kwargs are the same as were supplied in
    input, but each tensor in the original input.
    :param dump_graphs: Whether or not should also dump the internal graph representation of the
    original and compressed models in the .dot format into the log directory.
    :return: A controller for the compression algorithm (or algorithms, in which case the controller
    is an instance of CompositeCompressionController) and the model ready for compression parameter training wrapped
    as an object of NNCFNetwork."""

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
                                   ignored_scopes=ignored_scopes,
                                   target_scopes=target_scopes,
                                   scopes_without_shape_matching=scopes_without_shape_matching)

    should_init = resuming_state_dict is None
    composite_builder = PTCompositeCompressionAlgorithmBuilder(config, should_init=should_init)
    composite_builder.apply_to(compressed_model)

    compression_ctrl = compressed_model.commit_compression_changes()

    try:
        if resuming_state_dict is not None:
            load_state(compressed_model, resuming_state_dict, is_resume=True)
    finally:
        if dump_graphs and is_main_process() and composite_builder:
            if dummy_forward_fn is None:
                compressed_graph_builder = GraphBuilder(custom_forward_fn=
                                                        create_dummy_forward_fn(input_info_list,
                                                                                with_input_tracing=False))
            else:
                compressed_graph_builder = GraphBuilder(custom_forward_fn=dummy_forward_fn)

            graph = compressed_graph_builder.build_graph(compressed_model, compressed_model.get_tracing_context())
            graph.visualize_graph(osp.join(config.get("log_dir", "."), "compressed_graph.dot"))
    return compression_ctrl, compressed_model
