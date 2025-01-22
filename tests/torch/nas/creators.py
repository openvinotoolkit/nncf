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
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from examples.torch.common.models import efficient_net
from nncf import NNCFConfig
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_kernel import ElasticKernelHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.training.base_training import BNASTrainingController
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import (
    create_compressed_model_from_algo_names,
)
from nncf.experimental.torch.nas.bootstrapNAS.training.progressive_shrinking_controller import (
    ProgressiveShrinkingController,
)
from nncf.torch.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.torch.dynamic_graph.io_handling import FillerInputInfo
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.model_creation import create_nncf_network
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork
from tests.torch import test_models
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import get_empty_config
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.nas.descriptors import MultiElasticityTestDesc
from tests.torch.nas.helpers import move_model_to_cuda_if_available
from tests.torch.nas.models.synthetic import TwoConvModel
from tests.torch.nas.models.tcn import TCN
from tests.torch.nas.models.vgg_k7 import VGG11_K7
from tests.torch.test_models import mobilenet_v2
from tests.torch.test_models import mobilenet_v3_small


# TODO(nlyalyus) reduce number of creators and descriptors. create wrapper of TrainingAlgorithm  (ticket 81015)
def create_bootstrap_training_model_and_ctrl(
    model, nncf_config: NNCFConfig, register_bn_adapt: bool = True
) -> Tuple[NNCFNetwork, BNASTrainingController]:
    algo_name = nncf_config.get("bootstrapNAS", {}).get("training", {}).get("algorithm", "progressive_shrinking")
    nncf_network = create_nncf_network(model, nncf_config)
    if register_bn_adapt:
        register_bn_adaptation_init_args(nncf_config)
    ctrl, model = create_compressed_model_from_algo_names(nncf_network, nncf_config, [algo_name], False)

    # Default optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ctrl.set_training_lr_scheduler_args(optimizer, 1)
    return model, ctrl


def create_bnas_model_and_ctrl_by_test_desc(desc: MultiElasticityTestDesc):
    config = {
        "input_info": {"sample_size": desc.input_sizes},
        "bootstrapNAS": {"training": {"elasticity": {"depth": {"skipped_blocks": desc.blocks_to_skip}}}},
    }
    depth_config = config["bootstrapNAS"]["training"]["elasticity"]["depth"]
    if not desc.blocks_to_skip:
        del depth_config["skipped_blocks"]
    config["bootstrapNAS"]["training"]["elasticity"].update(desc.algo_params)

    nncf_config = NNCFConfig.from_dict(config)
    model = desc.model_creator()
    model.eval()
    move_model_to_cuda_if_available(model)
    model, training_ctrl = create_bootstrap_training_model_and_ctrl(model, nncf_config)
    return model, training_ctrl


def build_elastic_model_from_handler(nncf_network, handler):
    transformation_layout = PTTransformationLayout()
    commands = handler.get_transformation_commands()
    for command in commands:
        transformation_layout.register(command)
    transformer = PTModelTransformer(nncf_network)
    model = transformer.transform(transformation_layout)
    return model


NAS_MODEL_DESCS = {
    "resnet50": [test_models.ResNet50, [1, 3, 32, 32]],
    "resnet50_imagenet": [test_models.ResNet50, [1, 3, 224, 224]],
    "resnet18": [test_models.ResNet18, [1, 3, 32, 32]],
    "inception_v3": [partial(test_models.Inception3, num_classes=10), [1, 3, 299, 299]],
    "vgg11": [partial(test_models.VGG, "VGG11"), [1, 3, 32, 32]],
    "vgg11_k7": [VGG11_K7, [1, 3, 32, 32]],  # for testing elastic kernel
    "densenet_121": [test_models.DenseNet121, [1, 3, 32, 32]],
    "mobilenet_v2": [mobilenet_v2, [1, 3, 32, 32]],
    "mobilenet_v3_small": [mobilenet_v3_small, [1, 3, 32, 32]],
    "efficient_net_b0": [partial(efficient_net, model_name="efficientnet-b0", pretrained=False), [1, 3, 240, 240]],
    "squeezenet1_0": [test_models.squeezenet1_0, [1, 3, 32, 32]],
    "shufflenetv2": [partial(test_models.ShuffleNetV2, net_size=0.5), [1, 3, 32, 32]],
    "ssd_mobilenet": [test_models.ssd_mobilenet, [2, 3, 300, 300]],
    "resnext29_32x4d": [test_models.ResNeXt29_32x4d, [1, 3, 32, 32]],
    "pnasnetb": [test_models.PNASNetB, [1, 3, 32, 32]],
    "unet": [test_models.UNet, [1, 3, 360, 480]],
    "ssd_vgg": [test_models.ssd_vgg300, [2, 3, 300, 300]],
    "tcn": [partial(TCN, input_size=1, output_size=10, num_channels=[25] * 8, kernel_size=7, dropout=0.05), [1, 1, 3]],
}


def create_bootstrap_nas_training_algo(model_name) -> Tuple[NNCFNetwork, ProgressiveShrinkingController, Callable]:
    model = NAS_MODEL_DESCS[model_name][0]()
    nncf_config = get_empty_config(input_sample_sizes=NAS_MODEL_DESCS[model_name][1])
    nncf_config["bootstrapNAS"] = {"training": {"algorithm": "progressive_shrinking"}}
    nncf_config["input_info"][0].update({"filler": "random"})

    input_info = FillerInputInfo.from_nncf_config(nncf_config)
    dummy_forward = create_dummy_forward_fn(input_info)
    compressed_model, training_ctrl = create_bootstrap_training_model_and_ctrl(model, nncf_config)
    return compressed_model, training_ctrl, dummy_forward


def create_supernet(
    model_creator: Callable, input_sample_sizes: List[int], elasticity_params: Optional[Dict[str, Any]] = None
) -> Tuple[MultiElasticityHandler, NNCFNetwork]:
    params = {} if elasticity_params is None else elasticity_params
    config = get_empty_config(input_sample_sizes=input_sample_sizes)
    config["bootstrapNAS"] = {"training": {"elasticity": params}}
    model = model_creator()
    model, ctrl = create_bootstrap_training_model_and_ctrl(model, config)
    return ctrl.multi_elasticity_handler, model


def create_single_conv_kernel_supernet(
    kernel_size=5, out_channels=1, padding=2
) -> Tuple[ElasticKernelHandler, NNCFNetwork]:
    params = {"available_elasticity_dims": [ElasticityDim.KERNEL.value]}
    model_creator = partial(BasicConvTestModel, 1, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
    input_sample_sizes = [1, 1, kernel_size, kernel_size]
    multi_elasticity_handler, supernet = create_supernet(model_creator, input_sample_sizes, params)
    move_model_to_cuda_if_available(supernet)
    return multi_elasticity_handler.kernel_handler, supernet


def create_two_conv_width_supernet(elasticity_params=None, model=TwoConvModel):
    params = {"available_elasticity_dims": [ElasticityDim.WIDTH.value]}
    if elasticity_params is not None:
        params.update(elasticity_params)
    multi_elasticity_handler, supernet = create_supernet(model, model.INPUT_SIZE, params)
    move_model_to_cuda_if_available(supernet)
    return multi_elasticity_handler.width_handler, supernet
