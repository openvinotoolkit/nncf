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
from copy import deepcopy
from os import path as osp
from typing import Callable, Any, Tuple, List, Dict

from nncf.checkpoint_loading import load_state
from nncf.hw_config import HWConfigType
from torch.nn import Module

from nncf.compression_method_api import CompressionAlgorithmController, CompressionAlgorithmBuilder
from nncf.config import NNCFConfig
from nncf.debug import is_debug, set_debug_log_dir
from nncf.dynamic_graph.graph_builder import GraphBuilder, create_input_infos, create_dummy_forward_fn
from nncf.nncf_network import NNCFNetwork
from nncf.utils import is_main_process
from nncf.algo_selector import COMPRESSION_ALGORITHMS
from nncf.quantization.algo import QuantizerSetupType
from nncf.hw_config import HW_CONFIG_TYPE_TARGET_DEVICE_MAP

from nncf.nncf_logger import logger


def get_compression_algorithm(config):
    algorithm_key = config.get('algorithm', 'NoCompressionAlgorithmBuilder')
    logger.info("Creating compression algorithm: {}".format(algorithm_key))
    return COMPRESSION_ALGORITHMS.get(algorithm_key)


def create_compression_algorithm_builders(config: NNCFConfig,
                                          should_init: bool = True) -> List[CompressionAlgorithmBuilder]:
    compression_config_json_section = config.get('compression', {})
    compression_config_json_section = deepcopy(compression_config_json_section)

    hw_config_type = None
    quantizer_setup_type_str = config.get("quantizer_setup_type", "propagation_based")
    quantizer_setup_type = QuantizerSetupType.from_str(quantizer_setup_type_str)
    if quantizer_setup_type == QuantizerSetupType.PROPAGATION_BASED:
        target_device = config.get("target_device", "ANY")
        if target_device != 'NONE':
            hw_config_type = HWConfigType.from_str(HW_CONFIG_TYPE_TARGET_DEVICE_MAP[target_device])

    if isinstance(compression_config_json_section, dict):
        compression_config = NNCFConfig(compression_config_json_section)
        compression_config.register_extra_structs(config.get_all_extra_structs_for_copy())
        compression_config["hw_config_type"] = hw_config_type
        compression_config['quantizer_setup_type'] = quantizer_setup_type
        return [get_compression_algorithm(compression_config)(compression_config, should_init=should_init), ]
    retval = []
    for algo_config in compression_config_json_section:
        algo_config = NNCFConfig(algo_config)
        algo_config.register_extra_structs(config.get_all_extra_structs_for_copy())
        algo_config["hw_config_type"] = hw_config_type
        algo_config['quantizer_setup_type'] = quantizer_setup_type
        retval.append(get_compression_algorithm(algo_config)(algo_config, should_init=should_init))
    return retval


def create_compressed_model(model: Module, config: NNCFConfig,
                            resuming_state_dict: dict = None,
                            dummy_forward_fn: Callable[[Module], Any] = None,
                            wrap_inputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
                            dump_graphs=True,) \
    -> Tuple[CompressionAlgorithmController, NNCFNetwork]:
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

    if is_debug():
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
    compression_algo_builder_list = create_compression_algorithm_builders(config, should_init=should_init)

    for builder in compression_algo_builder_list:
        compressed_model = builder.apply_to(compressed_model)
    compression_ctrl = compressed_model.commit_compression_changes()

    if dump_graphs and is_main_process() and compression_algo_builder_list:
        if dummy_forward_fn is None:
            compressed_graph_builder = GraphBuilder(custom_forward_fn=
                                                    create_dummy_forward_fn(input_info_list,
                                                                            with_input_tracing=False))
        else:
            compressed_graph_builder = GraphBuilder(custom_forward_fn=dummy_forward_fn)

        graph = compressed_graph_builder.build_graph(compressed_model, compressed_model.get_tracing_context())
        graph.visualize_graph(osp.join(config.get("log_dir", "."), "compressed_graph.dot"))

    if resuming_state_dict is not None:
        load_state(compressed_model, resuming_state_dict, is_resume=True)

    return compression_ctrl, compressed_model
