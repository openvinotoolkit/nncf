# Copyright (c) 2021-2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from functools import partial
from typing import Callable, List, Optional, Type, Union

import pytest
import torch

from examples.torch.common.models import efficient_net
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlock
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlocks
from nncf.experimental.torch.search_building_blocks.search_blocks import GroupedBlockIDs
from nncf.experimental.torch.search_building_blocks.search_blocks import get_building_blocks
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import get_empty_config
from tests.torch.nas.helpers import move_model_to_cuda_if_available
from tests.torch.nas.test_elastic_depth import INCEPTION_INPUT_SIZE
from tests.torch.nas.test_elastic_depth import RESNET50_INPUT_SIZE
from tests.torch.test_models import MobileNetV2
from tests.torch.test_models import PNASNetB
from tests.torch.test_models import ResNet18
from tests.torch.test_models import ResNeXt29_32x4d
from tests.torch.test_models import squeezenet1_0
from tests.torch.test_models import ssd_mobilenet
from tests.torch.test_models.inceptionv3 import Inception3
from tests.torch.test_models.resnet import ResNet50


def check_blocks_and_groups(name, actual_blocks: BuildingBlocks, actual_group_dependent: GroupedBlockIDs):
    ref_file_dir = TEST_ROOT / "torch" / "data" / "search_building_block"
    ref_file_path = ref_file_dir.joinpath(name + ".json")
    if os.getenv("NNCF_TEST_REGEN_DOT") is not None:
        if not os.path.exists(ref_file_dir):
            os.makedirs(ref_file_dir)
        with ref_file_path.open("w", encoding="utf8") as f:
            actual_state = {
                "blocks": [block.get_state() for block in actual_blocks],
                "group_dependent": actual_group_dependent,
            }
            json.dump(actual_state, f, indent=4)

    with ref_file_path.open("r") as f:
        ref_state = json.load(f)
        ref_blocks = [BuildingBlock.from_state(state) for state in ref_state["blocks"]]
        ref_group_dependent = {int(k): v for k, v in ref_state["group_dependent"].items()}
        assert ref_blocks == actual_blocks
        assert ref_group_dependent == actual_group_dependent


class BuildingBlockParamsCase:
    def __init__(
        self,
        model_creator: Union[Type[torch.nn.Module], Callable[[], torch.nn.Module]],
        input_sizes: List[int],
        min_block_size: int = 5,
        max_block_size: int = 50,
        name: Optional[str] = None,
        hw_fused_ops: bool = True,
    ):
        self.model_creator = model_creator
        self.input_sizes = input_sizes
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        if not name and hasattr(self.model_creator, "__name__"):
            name = self.model_creator.__name__
        assert name, "Can't define name from the model (usually due to partial), please specify it explicitly"
        self.name = name.lower().replace(" ", "_")
        self.hw_fused_ops = hw_fused_ops

    def __str__(self):
        return self.name


LIST_BB_PARAMS_CASES = [
    BuildingBlockParamsCase(ResNet50, RESNET50_INPUT_SIZE, hw_fused_ops=False),
    BuildingBlockParamsCase(MobileNetV2, RESNET50_INPUT_SIZE),
    BuildingBlockParamsCase(Inception3, INCEPTION_INPUT_SIZE, min_block_size=23, name="Inception3_big_blocks"),
    BuildingBlockParamsCase(
        Inception3, INCEPTION_INPUT_SIZE, min_block_size=4, max_block_size=5, name="Inception3_small_blocks"
    ),
    BuildingBlockParamsCase(squeezenet1_0, RESNET50_INPUT_SIZE),
    BuildingBlockParamsCase(ResNeXt29_32x4d, [1, 3, 32, 32], hw_fused_ops=False),
    BuildingBlockParamsCase(PNASNetB, [1, 3, 32, 32]),
    BuildingBlockParamsCase(ssd_mobilenet, [2, 3, 300, 300], min_block_size=2, max_block_size=7),
    BuildingBlockParamsCase(
        partial(efficient_net, model_name="efficientnet-b0"),
        [10, 3, 240, 240],
        name="efficientnet-b0",
        min_block_size=2,
        max_block_size=7,
    ),
]


@pytest.mark.parametrize("desc", LIST_BB_PARAMS_CASES, ids=map(str, LIST_BB_PARAMS_CASES))
def test_building_block(desc: BuildingBlockParamsCase):
    model = desc.model_creator()
    move_model_to_cuda_if_available(model)
    nncf_config = get_empty_config(input_sample_sizes=desc.input_sizes)
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, nncf_config)

    ext_blocks, group_dependent = get_building_blocks(
        compressed_model,
        max_block_size=desc.max_block_size,
        min_block_size=desc.min_block_size,
        hw_fused_ops=desc.hw_fused_ops,
    )
    skipped_blocks = [eb.basic_block for eb in ext_blocks]
    check_blocks_and_groups(desc.name, skipped_blocks, group_dependent)


class SearchBBlockAlgoParamsCase:
    def __init__(self, min_block_size: int = 1, max_block_size: int = 100, hw_fused_ops: bool = False, ref_blocks=None):
        self.max_block_size = max_block_size
        self.min_block_size = min_block_size
        self.ref_blocks = [] if ref_blocks is None else ref_blocks
        self.hw_fused_ops = hw_fused_ops


@pytest.mark.parametrize(
    "algo_params",
    (
        (
            SearchBBlockAlgoParamsCase(max_block_size=5, ref_blocks=[]),
            SearchBBlockAlgoParamsCase(
                max_block_size=6,
                ref_blocks=[
                    BuildingBlock(
                        "ResNet/MaxPool2d[maxpool]/max_pool2d_0", "ResNet/Sequential[layer1]/BasicBlock[0]/relu_1"
                    ),
                    BuildingBlock(
                        "ResNet/Sequential[layer1]/BasicBlock[0]/relu_1",
                        "ResNet/Sequential[layer1]/BasicBlock[1]/relu_1",
                    ),
                    BuildingBlock(
                        "ResNet/Sequential[layer2]/BasicBlock[0]/relu_1",
                        "ResNet/Sequential[layer2]/BasicBlock[1]/relu_1",
                    ),
                    BuildingBlock(
                        "ResNet/Sequential[layer3]/BasicBlock[0]/relu_1",
                        "ResNet/Sequential[layer3]/BasicBlock[1]/relu_1",
                    ),
                    BuildingBlock(
                        "ResNet/Sequential[layer4]/BasicBlock[0]/relu_1",
                        "ResNet/Sequential[layer4]/BasicBlock[1]/relu_1",
                    ),
                ],
            ),
            SearchBBlockAlgoParamsCase(
                min_block_size=8,
                max_block_size=14,
                ref_blocks=[
                    BuildingBlock(
                        "ResNet/MaxPool2d[maxpool]/max_pool2d_0", "ResNet/Sequential[layer1]/BasicBlock[1]/relu_1"
                    )
                ],
            ),
            SearchBBlockAlgoParamsCase(
                min_block_size=7,
                max_block_size=7,
                ref_blocks=[
                    BuildingBlock(
                        "ResNet/Sequential[layer1]/BasicBlock[0]/__iadd___0",
                        "ResNet/Sequential[layer1]/BasicBlock[1]/relu_1",
                    ),
                    BuildingBlock(
                        "ResNet/Sequential[layer2]/BasicBlock[0]/__iadd___0",
                        "ResNet/Sequential[layer2]/BasicBlock[1]/relu_1",
                    ),
                    BuildingBlock(
                        "ResNet/Sequential[layer3]/BasicBlock[0]/__iadd___0",
                        "ResNet/Sequential[layer3]/BasicBlock[1]/relu_1",
                    ),
                    BuildingBlock(
                        "ResNet/Sequential[layer4]/BasicBlock[0]/__iadd___0",
                        "ResNet/Sequential[layer4]/BasicBlock[1]/relu_1",
                    ),
                ],
            ),
        )
    ),
)
def test_building_block_algo_param(algo_params: SearchBBlockAlgoParamsCase):
    model = ResNet18()
    move_model_to_cuda_if_available(model)
    nncf_config = get_empty_config(input_sample_sizes=RESNET50_INPUT_SIZE)
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, nncf_config)

    ext_blocks, _ = get_building_blocks(
        compressed_model,
        min_block_size=algo_params.min_block_size,
        max_block_size=algo_params.max_block_size,
    )
    blocks = [eb.basic_block for eb in ext_blocks]
    assert blocks == algo_params.ref_blocks
