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

import json
import os
from functools import partial
from typing import Callable, Dict, List, Optional, Set, Union

import numpy as np
import pytest
import torch
from torch import nn
from transformers import AutoModelForAudioClassification
from transformers import AutoModelForImageClassification
from transformers import AutoModelForQuestionAnswering
from transformers import BertConfig
from transformers import SwinConfig
from transformers import ViTConfig
from transformers import Wav2Vec2Config

from nncf.experimental.torch.search_building_blocks.search_blocks import BlockFilteringStrategy
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlockType
from nncf.experimental.torch.search_building_blocks.search_blocks import ExtendedBuildingBlock
from nncf.experimental.torch.search_building_blocks.search_blocks import ExtendedBuildingBlocks
from nncf.experimental.torch.search_building_blocks.search_blocks import get_building_blocks
from nncf.experimental.torch.search_building_blocks.search_blocks import get_indexes_of_overlapping_blocks_min
from nncf.experimental.torch.search_building_blocks.search_blocks import get_indexes_of_overlapping_blocks_seq
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import get_empty_config
from tests.torch.nas.helpers import move_model_to_cuda_if_available


def check_extended_blocks(name, actual_blocks: ExtendedBuildingBlocks):
    ref_file_dir = TEST_ROOT / "torch" / "data" / "search_building_block"
    ref_file_path = ref_file_dir.joinpath(name + ".json")
    if os.getenv("NNCF_TEST_REGEN_DOT") is not None:
        if not os.path.exists(ref_file_dir):
            os.makedirs(ref_file_dir)
        with ref_file_path.open("w", encoding="utf8") as f:
            actual_state = [block.get_state() for block in actual_blocks]
            json.dump(actual_state, f, indent=4, sort_keys=True)

    with ref_file_path.open("r") as f:
        ref_state = json.load(f)
        ref_blocks = [ExtendedBuildingBlock.from_state(state) for state in ref_state]
        assert ref_blocks == actual_blocks, (
            "Blocks are different, set NNCF_TEST_REGEN_DOT=1 to override references "
            "and compare the difference using `git diff` or other tools for comparison"
        )


class TransformerSearchBBlockParamsCase:
    def __init__(self, name: str, input_info: Union[List, Dict], model_creator: Callable[[], nn.Module]):
        self.input_info = input_info
        self.model_creator = model_creator
        self.name = name.lower().replace(" ", "_")


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(768, 768)
        self.k = nn.Linear(768, 768)
        self.v = nn.Linear(768, 768)
        self.o = nn.Linear(768, 768)
        self.sm = nn.Softmax()

    def forward(self, x):
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        k = k.view(-1, 12, 64).permute(1, 0, 2)
        q = q.view(-1, 12, 64).permute(1, 2, 0)
        v = v.view(-1, 12, 64).permute(1, 0, 2)
        x = self.sm(torch.matmul(k, q)) / np.sqrt(1 / 384)
        x = torch.matmul(x, v)
        x = x.permute(1, 0, 2).contiguous().view(-1, 768)
        return self.o(x)


LIST_CASES = [
    TransformerSearchBBlockParamsCase(
        name="BERT",
        input_info=[dict(sample_size=[1, 10], type="long")],
        model_creator=partial(AutoModelForQuestionAnswering.from_config, BertConfig()),
    ),
    TransformerSearchBBlockParamsCase(
        name="ViT",
        input_info=dict(sample_size=[1, 3, 224, 224]),
        model_creator=partial(AutoModelForImageClassification.from_config, ViTConfig()),
    ),
    TransformerSearchBBlockParamsCase(
        name="wave2vec 2.0",
        input_info=dict(sample_size=[1, 400]),
        model_creator=partial(AutoModelForAudioClassification.from_config, Wav2Vec2Config()),
    ),
    TransformerSearchBBlockParamsCase(
        name="SWIN MS",
        input_info=dict(sample_size=[1, 3, 224, 224]),
        model_creator=partial(AutoModelForImageClassification.from_config, SwinConfig()),
    ),
    TransformerSearchBBlockParamsCase(
        name="one MHSA",
        input_info=dict(sample_size=[384, 768]),
        model_creator=SelfAttention,
    ),
]


@pytest.fixture(name="desc", scope="function", params=LIST_CASES, ids=map(lambda x: x.name, LIST_CASES))
def fixture_transformer_search_params_desc(request):
    return request.param


def test_transformer_building_blocks(desc: TransformerSearchBBlockParamsCase):
    model = desc.model_creator()
    move_model_to_cuda_if_available(model)
    nncf_config = get_empty_config(input_info=desc.input_info)
    nncf_model, _ = create_compressed_model_and_algo_for_test(model, nncf_config)

    ext_blocks, _ = get_building_blocks(
        nncf_model,
        target_block_types=[BuildingBlockType.MHSA, BuildingBlockType.FF],
        block_filter_strategy=BlockFilteringStrategy.KEEP_SMALL,
        hw_fused_ops=True,
    )
    check_extended_blocks(desc.name, ext_blocks)


class FilterBlockTestDesc:
    def __init__(
        self,
        start_ids: List[int],
        end_ids: List[int],
        overlapping_blocks_ids_min: Optional[Set[int]] = None,
        overlapping_blocks_ids_seq: Optional[Set[int]] = None,
        num_ops_in_block: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        self.start_ids = start_ids
        self.end_ids = end_ids
        self.overlapping_blocks_ids_min = overlapping_blocks_ids_min
        if self.overlapping_blocks_ids_min is None:
            self.overlapping_blocks_ids_min = set()
        self.overlapping_blocks_ids_seq = overlapping_blocks_ids_seq
        if self.overlapping_blocks_ids_seq is None:
            self.overlapping_blocks_ids_seq = self.overlapping_blocks_ids_min
        self.num_ops_in_block = num_ops_in_block
        if self.num_ops_in_block is None:
            self.num_ops_in_block = [e - s for s, e in zip(self.start_ids, self.end_ids)]
        self.name = name
        if self.name is None:
            self.name = "__".join(f"{s}:{e}" for s, e in zip(self.start_ids, self.end_ids))

    def __str__(self):
        return self.name


LIST_FILTER_BLOCK_DESCS = [
    FilterBlockTestDesc(
        name="empty",
        start_ids=[],
        end_ids=[],
    ),
    FilterBlockTestDesc(
        start_ids=[1],
        end_ids=[2],
    ),
    FilterBlockTestDesc(
        start_ids=[1, 2, 3, 4],
        end_ids=[2, 3, 4, 5],
    ),
    FilterBlockTestDesc(
        start_ids=[1, 2, 3, 4],
        end_ids=[5, 5, 5, 5],
        overlapping_blocks_ids_min={0, 1, 2},
        overlapping_blocks_ids_seq={1, 2, 3},
    ),
    FilterBlockTestDesc(
        start_ids=[1, 1, 1, 1],
        end_ids=[2, 3, 4, 5],
        overlapping_blocks_ids_min={1, 2, 3},
    ),
    FilterBlockTestDesc(start_ids=[1, 1, 2, 2], end_ids=[2, 3, 3, 4], overlapping_blocks_ids_min={1, 3}),
    FilterBlockTestDesc(
        start_ids=[1, 1, 2, 2],
        end_ids=[4, 3, 3, 4],
        overlapping_blocks_ids_min={0, 1, 3},
        overlapping_blocks_ids_seq={0, 2, 3},
    ),
    FilterBlockTestDesc(start_ids=[1, 2, 2, 1], end_ids=[4, 3, 4, 3], overlapping_blocks_ids_min={0, 2, 3}),
    FilterBlockTestDesc(
        start_ids=[1, 3, 3, 4, 5, 10, 11],
        end_ids=[4, 5, 6, 7, 6, 14, 12],
        overlapping_blocks_ids_min={0, 2, 3, 5},
        overlapping_blocks_ids_seq={1, 2, 4, 6},
    ),
    FilterBlockTestDesc(
        start_ids=[3, 10, 3, 5, 11, 1, 4],
        end_ids=[6, 14, 5, 6, 12, 4, 7],
        overlapping_blocks_ids_min={0, 1, 5, 6},
        overlapping_blocks_ids_seq={0, 4, 5, 6},
    ),
    FilterBlockTestDesc(start_ids=[1, 2, 3, 4], end_ids=[5, 4, 6, 9], overlapping_blocks_ids_min={0, 2}),
    FilterBlockTestDesc(start_ids=[1, 3, 2, 4], end_ids=[5, 6, 4, 9], overlapping_blocks_ids_min={0, 1}),
    FilterBlockTestDesc(
        name="non_standard_num_ops",
        start_ids=[1, 2, 2, 1],
        end_ids=[4, 3, 4, 3],
        num_ops_in_block=[1, 10, 2, 11],
        overlapping_blocks_ids_min={1, 2, 3},
        overlapping_blocks_ids_seq={0, 2, 3},
    ),
    FilterBlockTestDesc(
        name="SSD_mobilenet",
        start_ids=[45, 48, 45, 51, 48, 54, 51, 57, 54, 60, 57, 63, 60, 66, 63],
        end_ids=[51, 54, 54, 57, 57, 60, 60, 63, 63, 66, 66, 69, 69, 72, 72],
        num_ops_in_block=[6, 6, 9, 6, 9, 6, 9, 6, 9, 6, 9, 6, 9, 6, 9],
        overlapping_blocks_ids_min={1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14},
    ),
    FilterBlockTestDesc(
        name="efficient-net-b0",
        start_ids=[
            10,
            28,
            46,
            37,
            65,
            83,
            74,
            102,
            120,
            111,
            139,
            130,
            158,
            176,
            167,
            195,
            186,
            214,
            232,
            223,
            251,
            242,
            270,
            261,
            289,
        ],
        end_ids=[
            17,
            35,
            53,
            56,
            72,
            90,
            93,
            109,
            127,
            130,
            146,
            149,
            165,
            183,
            186,
            202,
            205,
            221,
            239,
            242,
            258,
            261,
            277,
            280,
            296,
        ],
        num_ops_in_block=[7, 7, 7, 19, 7, 7, 19, 7, 7, 19, 7, 19, 7, 7, 19, 7, 19, 7, 7, 19, 7, 19, 7, 19, 7],
        overlapping_blocks_ids_min={3, 6, 9, 11, 14, 16, 19, 21, 23},
        overlapping_blocks_ids_seq={3, 6, 8, 10, 13, 15, 18, 20, 22},
    ),
    FilterBlockTestDesc(
        name="mobilenet-v3",
        start_ids=[6, 6, 22, 37, 51, 51, 45, 66, 66, 60, 81, 81, 95, 95, 89, 110, 124, 124, 118, 139, 139, 133],
        end_ids=[12, 14, 31, 43, 57, 57, 60, 72, 72, 75, 87, 87, 101, 101, 104, 116, 130, 130, 133, 145, 145, 148],
        num_ops_in_block=[6, 8, 9, 6, 6, 6, 15, 6, 6, 15, 6, 6, 6, 6, 15, 6, 6, 6, 15, 6, 6, 15],
        overlapping_blocks_ids_min={1, 5, 6, 8, 9, 11, 13, 14, 17, 18, 20, 21},
        overlapping_blocks_ids_seq={1, 4, 5, 7, 8, 10, 12, 14, 16, 17, 19, 20},
    ),
    FilterBlockTestDesc(
        name="resnet50_unfused",
        start_ids=[5, 17, 16, 16, 27, 26, 16, 26, 16],
        end_ids=[10, 23, 25, 26, 33, 35, 35, 36, 36],
        num_ops_in_block=[5, 6, 9, 10, 6, 9, 19, 10, 20],
        overlapping_blocks_ids_min={2, 3, 5, 6, 7, 8},
        overlapping_blocks_ids_seq={1, 2, 4, 6, 7, 8},
    ),
    FilterBlockTestDesc(
        name="part_of_BERT",
        start_ids=[10, 9, 10, 9, 32, 33, 32, 38, 39, 38, 39, 61, 62, 61, 67, 68, 67, 68],
        end_ids=[32, 32, 33, 33, 38, 39, 39, 61, 61, 62, 62, 67, 68, 68, 90, 90, 91, 91],
        num_ops_in_block=[21, 23, 22, 24, 6, 5, 7, 23, 21, 24, 22, 6, 5, 7, 23, 21, 24, 22],
        overlapping_blocks_ids_min={1, 2, 3, 4, 6, 7, 9, 10, 11, 13, 14, 16, 17},
        overlapping_blocks_ids_seq={1, 2, 3, 5, 6, 8, 9, 10, 12, 13, 15, 16, 17},
    ),
]


@pytest.fixture(
    name="filter_blocks_desc", scope="function", params=LIST_FILTER_BLOCK_DESCS, ids=map(str, LIST_FILTER_BLOCK_DESCS)
)
def fixture_filter_blocks_desc(request) -> FilterBlockTestDesc:
    return request.param


def test_filter_with_keeping_small(filter_blocks_desc: FilterBlockTestDesc):
    actual_indexes_of_overlapping_blocks = get_indexes_of_overlapping_blocks_min(
        filter_blocks_desc.start_ids,
        filter_blocks_desc.end_ids,
        filter_blocks_desc.num_ops_in_block,
    )
    assert actual_indexes_of_overlapping_blocks == filter_blocks_desc.overlapping_blocks_ids_min

    actual_indexes_of_overlapping_blocks = get_indexes_of_overlapping_blocks_seq(
        filter_blocks_desc.start_ids,
        filter_blocks_desc.end_ids,
        filter_blocks_desc.num_ops_in_block,
    )
    assert actual_indexes_of_overlapping_blocks == filter_blocks_desc.overlapping_blocks_ids_seq
