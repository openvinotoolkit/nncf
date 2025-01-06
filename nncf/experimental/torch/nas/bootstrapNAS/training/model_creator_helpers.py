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

from os import path as osp
from typing import Any, Dict, List, Optional, Tuple

from nncf.common.compression import BaseCompressionAlgorithmController as BaseController
from nncf.common.utils.debug import set_debug_log_dir
from nncf.config import NNCFConfig
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.model_creation import create_compression_algorithm_builder_from_algo_names
from nncf.torch.model_creation import synchronize_all_processes_in_distributed_mode
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import is_main_process


def create_compressed_model_from_algo_names(
    nncf_network: NNCFNetwork, config: NNCFConfig, algo_names: List[str], dump_graphs: bool = True
) -> Tuple[BaseController, NNCFNetwork]:
    """
    The main function used to produce a model ready for compression fine-tuning from empty NNCFNetwork,
    a configuration object and a list of compression algorithm names.

    :param nncf_network: empty NNCFNetwork, that just wrapped original PyTorch model.
    :param config: A configuration object used to determine the exact compression modifications to be applied
    to the model
    :param algo_names: list of compression algorithm names.
    :param dump_graphs: Whether or not should also dump the internal graph representation of the
    original and compressed models in the .dot format into the log directory.
    :return: A controller for the compression algorithm (or algorithms, in which case the controller
    is an instance of CompositeCompressionController) and the model ready for compression parameter training wrapped
    as an object of NNCFNetwork.
    """
    set_debug_log_dir(config.get("log_dir", ""))

    if dump_graphs:
        original_model_graph = nncf_network.nncf.get_original_graph()
        original_model_graph.visualize_graph(osp.join(config.get("log_dir", ""), "original_graph.dot"))

    builder = create_compression_algorithm_builder_from_algo_names(algo_names, config, should_init=True)

    compressed_model = builder.apply_to(nncf_network)
    compression_ctrl = builder.build_controller(compressed_model)

    # Required to ensure that the model leaving create_compressed_model has correct compressed graph.
    # In particular, this is currently required for correct functioning of RNNs.
    compressed_model.nncf.rebuild_graph()

    if dump_graphs and is_main_process():
        original_model_graph = compressed_model.nncf.get_graph()
        original_model_graph.visualize_graph(osp.join(config.get("log_dir", ""), "compressed_graph.dot"))

    synchronize_all_processes_in_distributed_mode()
    return compression_ctrl, compressed_model


def resume_compression_algorithm_builder(
    compression_state: Dict[str, Any], config: Optional[NNCFConfig] = None
) -> PTCompressionAlgorithmBuilder:
    """
    Resume compression builder from its state.

    :param compression_state: representation of the entire compression state to unambiguously restore
    the compressed model. Includes builder and controller states.
    :param config: A configuration object used to determine the exact compression modifications to be applied
    to the model. NNCFConfig is required for resume from checkpoint, because currently CompressionBuilder takes
    NNCFConfig to get some params, like BNAdaptInitArgs. It's needed not only on init. e.g.
    BNAdaptInitArgs.train_loader is used by ProgressiveShrinkingAlgorithm during the training.
    :return: resumed compression builder
    """
    if config is None:
        config = NNCFConfig()

    builder_state = compression_state[BaseController.BUILDER_STATE]
    algo_names = list(builder_state)
    builder = create_compression_algorithm_builder_from_algo_names(algo_names, config, should_init=False)
    builder.load_state(builder_state)
    return builder


def resume_compression_from_state(
    nncf_network: NNCFNetwork, compression_state: Dict[str, Any], config: Optional[NNCFConfig] = None
) -> Tuple[NNCFNetwork, PTCompressionAlgorithmController]:
    """
    Resumes compression model

    :param nncf_network: empty NNCFNetwork, that just wrapped original PyTorch model.
    :param compression_state: representation of the entire compression state to unambiguously restore
    the compressed model. Includes builder and controller states.
    :param config: is needed for overriding state or for passing extra structs only, like BNAdaptInitArgs required
    during training, like in NAS.
    :return: the model ready for compression parameter training wrapped as an object of NNCFNetwork and
    a controller for the compression algorithm.
    """
    builder = resume_compression_algorithm_builder(compression_state, config)
    model = builder.apply_to(nncf_network)
    ctrl = builder.build_controller(nncf_network)
    ctrl.load_state(compression_state[BaseController.CONTROLLER_STATE])
    return model, ctrl
