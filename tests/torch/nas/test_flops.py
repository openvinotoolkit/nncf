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

import pytest
from torchvision.models import resnet50

from examples.torch.common.models.classification.mobilenet_v2_cifar10 import mobilenet_v2_cifar10
from examples.torch.common.models.classification.resnet_cifar10 import resnet50_cifar10
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from tests.torch.nas.creators import create_bnas_model_and_ctrl_by_test_desc
from tests.torch.nas.descriptors import THREE_CONV_TEST_DESC
from tests.torch.nas.descriptors import ModelStats
from tests.torch.nas.descriptors import MultiElasticityTestDesc
from tests.torch.nas.descriptors import RefModelStats
from tests.torch.nas.test_elastic_depth import DepthBasicConvTestModel
from tests.torch.test_models import DenseNet121
from tests.torch.test_models import mobilenet_v2

RESNET50_BLOCK_TO_SKIP = [
    [
        "ResNet/Sequential[layer1]/Bottleneck[1]/ReLU[relu]/relu__2",
        "ResNet/Sequential[layer1]/Bottleneck[2]/ReLU[relu]/relu__2",
    ],
]

MOBILENET_V2_BLOCKS_TO_SKIP = [
    [
        "MobileNetV2/Sequential[features]/InvertedResidual[2]/Sequential[conv]/NNCFConv2d[2]/conv2d_0",
        "MobileNetV2/Sequential[features]/InvertedResidual[3]/__add___0",
    ],
    [
        "MobileNetV2/Sequential[features]/InvertedResidual[4]/Sequential[conv]/NNCFConv2d[2]/conv2d_0",
        "MobileNetV2/Sequential[features]/InvertedResidual[5]/__add___0",
    ],
]

LIST_OF_ME_DESCS = [
    MultiElasticityTestDesc(
        model_creator=resnet50_cifar10,
        ref_model_stats=RefModelStats(
            supernet=ModelStats(651_599_872, 23_467_712),
            kernel_stage=ModelStats(651_599_872, 23_467_712),
            depth_stage=ModelStats(615_948_288, 23_398_080),
            width_stage=ModelStats(22_717_056, 174_240),
        ),
        blocks_to_skip=RESNET50_BLOCK_TO_SKIP,
    ),
    MultiElasticityTestDesc(
        name="resnet50_tv",
        model_creator=partial(resnet50, num_classes=10),
        ref_model_stats=RefModelStats(
            supernet=ModelStats(166_862_848, 23_475_392),
            kernel_stage=ModelStats(162_930_688, 23_467_712),
            depth_stage=ModelStats(154_017_792, 23_398_080),
            width_stage=ModelStats(5_679_744, 174_240),
        ),
        blocks_to_skip=RESNET50_BLOCK_TO_SKIP,
    ),
    MultiElasticityTestDesc(
        name="mobilenet_tv",
        model_creator=partial(mobilenet_v2, num_classes=10),
        ref_model_stats=RefModelStats(
            supernet=ModelStats(12_249_856, 2_202_560),
            kernel_stage=ModelStats(12_249_856, 2_202_560),
            depth_stage=ModelStats(10_750_720, 2_180_336),
            width_stage=ModelStats(1_717_376, 35_728),
        ),
        blocks_to_skip=MOBILENET_V2_BLOCKS_TO_SKIP,
    ),
    MultiElasticityTestDesc(
        name="mobilenet_tv_imagenet",
        input_sizes=[1, 3, 224, 224],
        model_creator=partial(mobilenet_v2, num_classes=1000),
        ref_model_stats=RefModelStats(
            supernet=ModelStats(601_548_544, 3_469_760),
            kernel_stage=ModelStats(601_548_544, 3_469_760),
            depth_stage=ModelStats(528_090_880, 3_447_536),
            width_stage=ModelStats(84_184_064, 67_408),
        ),
        blocks_to_skip=MOBILENET_V2_BLOCKS_TO_SKIP,
    ),
    MultiElasticityTestDesc(
        model_creator=mobilenet_v2_cifar10,
        ref_model_stats=RefModelStats(
            supernet=ModelStats(175_952_896, 2_202_560),
            kernel_stage=ModelStats(175_952_896, 2_202_560),
            depth_stage=ModelStats(151_966_720, 2_180_336),
            width_stage=ModelStats(14_099_072, 35_728),
        ),
        blocks_to_skip=MOBILENET_V2_BLOCKS_TO_SKIP,
    ),
    MultiElasticityTestDesc(
        model_creator=DenseNet121,
        name="densenet_121",
        ref_model_stats=RefModelStats(
            supernet=ModelStats(1_776_701_440, 6_872_768),
            kernel_stage=ModelStats(1_776_701_440, 6_872_768),
            depth_stage=ModelStats(1_223_053_312, 6_602_432),
            width_stage=ModelStats(358_165_120, 1_134_752),
        ),
    ),
    THREE_CONV_TEST_DESC,
    MultiElasticityTestDesc(
        model_creator=DepthBasicConvTestModel,
        ref_model_stats=RefModelStats(
            supernet=ModelStats(317_250, 705),
            kernel_stage=ModelStats(122_850, 273),
            depth_stage=ModelStats(86_400, 192),
            width_stage=ModelStats(12_600, 28),
        ),
        blocks_to_skip=[
            [
                "DepthBasicConvTestModel/Sequential[branch_with_blocks]/NNCFConv2d[conv0]/conv2d_0",
                "DepthBasicConvTestModel/Sequential[branch_with_blocks]/NNCFConv2d[conv1]/conv2d_0",
            ]
        ],
        input_sizes=DepthBasicConvTestModel.INPUT_SIZE,
        algo_params={"width": {"min_width": 1, "width_step": 1}},
    ),
]


@pytest.mark.parametrize("desc", LIST_OF_ME_DESCS, ids=map(str, LIST_OF_ME_DESCS))
def test_multi_elasticity_flops(desc: MultiElasticityTestDesc):
    model, ctrl = create_bnas_model_and_ctrl_by_test_desc(desc)
    ref_model_stats = desc.ref_model_stats
    multi_elasticity_handler = ctrl.multi_elasticity_handler

    assert multi_elasticity_handler.count_flops_and_weights_for_active_subnet() == ref_model_stats.supernet
    model.nncf.do_dummy_forward()

    multi_elasticity_handler.disable_all()
    multi_elasticity_handler.enable_elasticity(ElasticityDim.KERNEL)
    multi_elasticity_handler.activate_minimum_subnet()
    assert multi_elasticity_handler.count_flops_and_weights_for_active_subnet() == ref_model_stats.kernel_stage
    model.nncf.do_dummy_forward()

    multi_elasticity_handler.enable_elasticity(ElasticityDim.DEPTH)
    multi_elasticity_handler.activate_minimum_subnet()

    assert multi_elasticity_handler.count_flops_and_weights_for_active_subnet() == ref_model_stats.depth_stage
    model.nncf.do_dummy_forward()

    multi_elasticity_handler.enable_elasticity(ElasticityDim.WIDTH)
    multi_elasticity_handler.activate_minimum_subnet()
    assert multi_elasticity_handler.count_flops_and_weights_for_active_subnet() == ref_model_stats.width_stage
    model.nncf.do_dummy_forward()
