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
from typing import Any, Dict, List, NamedTuple, Optional

import pytest

from examples.torch.common.models.classification.mobilenet_v2_cifar10 import MobileNetV2
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from tests.torch.helpers import BasicConvTestModel
from tests.torch.nas.descriptors import ElasticityDesc
from tests.torch.nas.descriptors import WidthElasticityDesc
from tests.torch.nas.models.synthetic import TwoConvModel
from tests.torch.nas.test_elastic_depth import RESNET50_INPUT_SIZE
from tests.torch.nas.test_elastic_depth import DepthBasicConvTestModel
from tests.torch.nas.test_state import COMMON_DEPTH_BASIC_DESC
from tests.torch.nas.test_state import COMMON_DEPTH_SUPERNET_DESC
from tests.torch.test_models import ResNet18

KERNEL_SIZE_AND_SEARCH_SPACE = [(5, [5, 3]), (7, [7, 5, 3]), (1, [1])]

LIST_KERNEL_SS_DESCS = [
    ElasticityDesc(
        ElasticityDim.KERNEL,
        model_cls=partial(BasicConvTestModel, 1, 1, kernel_size, padding=2),
        name=f"kernel_{kernel_size}_{search_space}",
        input_size=[1, 1, kernel_size, kernel_size],
        ref_search_space=[search_space],
    )
    for kernel_size, search_space in KERNEL_SIZE_AND_SEARCH_SPACE
]


class WidthSearchSpaceParams(NamedTuple):
    max_out_channels: int
    params: Optional[Dict[str, Any]]
    list_output_channels: List[int]
    width_indicator: int = -1

    def __str__(self):
        result = f"max:{self.max_out_channels}"
        if self.params:
            result += f"__params:{self.params}"
        return result


LIST_WSS_PARAMS = [
    WidthSearchSpaceParams(32, None, [32]),
    WidthSearchSpaceParams(16, None, [16]),
    WidthSearchSpaceParams(64, None, [64, 32]),
    WidthSearchSpaceParams(64, None, [64], width_indicator=1),
    WidthSearchSpaceParams(144, None, [144, 112, 80, 48]),
    WidthSearchSpaceParams(144, None, [144], width_indicator=1),
    WidthSearchSpaceParams(64, {"width_multipliers": [0.25, 0.5, 0.125]}, ([64]), width_indicator=1),
    WidthSearchSpaceParams(64, {"width_multipliers": [0.25, 0.5, 0.125], "min_width": 8}, [64, 32, 16, 8]),
    WidthSearchSpaceParams(64, {"width_multipliers": [0, 1, 0.25, 0.5, 0.125], "min_width": 8}, [64, 32, 16, 8]),
    WidthSearchSpaceParams(64, {"width_multipliers": [0, 1, 0.25, 0.5, 0.125], "min_width": 8}, [64, 32, 16, 8]),
    WidthSearchSpaceParams(32, {"min_width": 16}, [32]),
    WidthSearchSpaceParams(32, {"min_width": 16, "width_step": 16}, [32, 16]),
    WidthSearchSpaceParams(64, {"width_step": 16, "max_num_widths": 3}, [64, 48, 32]),
    WidthSearchSpaceParams(64, {"min_width": 64}, [64]),
    WidthSearchSpaceParams(64, {"max_num_widths": 1}, [64]),
    WidthSearchSpaceParams(64, {"max_num_widths": 1, "width_multipliers": [1, 0.5, 0.25]}, [64]),
    WidthSearchSpaceParams(64, {"max_num_widths": 2}, [64, 32]),
]

LIST_WIDTH_SS_DESCS = [
    WidthElasticityDesc(
        ElasticityDesc(
            ElasticityDim.WIDTH,
            model_cls=partial(TwoConvModel, in_channels=1, out_channels=wss_params.max_out_channels, kernel_size=5),
            params=wss_params.params,
            name=f"width_{wss_params}",
            input_size=[1, 1, 5, 5],
            ref_search_space={0: wss_params.list_output_channels},
        ),
        width_num_params_indicator=wss_params.width_indicator,
    )
    for wss_params in LIST_WSS_PARAMS
]

LIST_SEARCH_SPACE_DESCS = [
    COMMON_DEPTH_BASIC_DESC,
    ElasticityDesc(
        ElasticityDim.DEPTH,
        model_cls=DepthBasicConvTestModel,
        params={"min_block_size": 1, "hw_fused_ops": False},
        ref_search_space=[[0], []],
    ),
    COMMON_DEPTH_SUPERNET_DESC,
    ElasticityDesc(
        ElasticityDim.DEPTH,
        model_cls=ResNet18,
        input_size=RESNET50_INPUT_SIZE,
        ref_search_space=[
            [1],
            [2],
            [3],
            [4],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 3],
            [2, 4],
            [3, 4],
            [1, 2, 3],
            [1, 2, 4],
            [1, 3, 4],
            [2, 3, 4],
            [1, 2, 3, 4],
            [],
        ],
    ),
    ElasticityDesc(
        ElasticityDim.DEPTH,
        model_cls=MobileNetV2,
        input_size=RESNET50_INPUT_SIZE,
        ref_search_space=[
            [0],
            [2],
            [5],
            [7],
            [9],
            [0, 2],
            [0, 5],
            [0, 7],
            [0, 9],
            [2, 5],
            [2, 7],
            [2, 9],
            [5, 7],
            [5, 9],
            [7, 9],
            [0, 2, 5],
            [0, 2, 7],
            [0, 2, 9],
            [0, 5, 7],
            [0, 5, 9],
            [0, 7, 9],
            [2, 5, 7],
            [2, 5, 9],
            [2, 7, 9],
            [5, 7, 9],
            [0, 2, 5, 7],
            [0, 2, 5, 9],
            [0, 2, 7, 9],
            [0, 5, 7, 9],
            [2, 5, 7, 9],
            [0, 2, 5, 7, 9],
            [],
        ],
    ),
    ElasticityDesc(
        ElasticityDim.DEPTH,
        model_cls=MobileNetV2,
        name="MobileNetV2_auto",
        params={"min_block_size": 10},
        input_size=RESNET50_INPUT_SIZE,
        ref_search_space=[
            [0],
            [1],
            [2],
            [3],
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [0, 1, 2, 3],
            [],
        ],
    ),
    *LIST_KERNEL_SS_DESCS,
    *LIST_WIDTH_SS_DESCS,
]


@pytest.mark.parametrize("desc", LIST_SEARCH_SPACE_DESCS, ids=map(str, LIST_SEARCH_SPACE_DESCS))
def test_elastic_search_space(desc: ElasticityDesc):
    handler, _ = desc.build_handler()
    actual_search_space = handler.get_search_space()
    assert actual_search_space == desc.ref_search_space
