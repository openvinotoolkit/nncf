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
from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import numpy as np
import pytest
import torch
from torch import nn

from nncf import NNCFConfig
from nncf.torch.model_creation import create_nncf_network
from tests.cross_fw.shared.nx_graph import compare_nx_graph_with_reference
from tests.torch.test_compressed_graph import GeneralModelDesc
from tests.torch.test_compressed_graph import IModelDesc
from tests.torch.test_compressed_graph import get_full_path_to_the_graph
from tests.torch.test_models.swin import SwinTransformerBlock

# isort: off
from nncf.experimental.common.pruning.block_hierarchy import BlockHierarchy
from nncf.experimental.common.pruning.nodes_grouping import PruningBlock
from nncf.experimental.common.pruning.nodes_grouping import PruningGroup
from nncf.experimental.common.pruning.nodes_grouping import get_pruning_groups
from nncf.experimental.common.pruning.nodes_grouping import select_largest_groups
from nncf.experimental.common.pruning.propagation_data import ConsumerInfo
from nncf.experimental.common.pruning.propagation_data import ProducerInfo
from nncf.experimental.torch.pruning.operations import PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES

# NNCF Torch should be imported before transformers in order to patch all operations before they
# added to some global vars,  otherwise test may fail with some error (e.g. IndexError: list index out of range).
from transformers import AutoModelForAudioClassification
from transformers import AutoModelForImageClassification
from transformers import AutoModelForQuestionAnswering
from transformers import AutoModelForSequenceClassification
from transformers import BertConfig
from transformers import CLIPVisionConfig
from transformers import CLIPVisionModel
from transformers import DistilBertConfig
from transformers import MobileBertConfig
from transformers import RobertaConfig
from transformers import SwinConfig
from transformers import ViTConfig
from transformers import Wav2Vec2Config
from transformers import AutoConfig


class SelfAttention(nn.Module):
    INPUT_SAMPLE_SIZES = [384, 768]

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


class DiffNumBranchesOnJoining(nn.Module):
    INPUT_SAMPLE_SIZES = [1, 1]

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(1, 6)
        self.k = nn.Linear(1, 6)

    def forward(self, x):
        k = self.k(x)  # [1, 6]
        q = self.q(x)  # [1, 6]
        k = k.view(1, 2, 3).permute(0, 2, 1)  # [1, 3, 2]
        q_reshaped = q.view(1, 2, 3)  # [1, 2, 3]
        o1 = torch.matmul(k, q_reshaped)
        return o1, q


class ReshapeReshape1Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(3, 60)
        self.final_2 = nn.Linear(2, 5)
        self.final_4 = nn.Linear(4, 3)

    def forward(self, x):
        base = self.base(x)
        reshape4 = base.view(1, 4, 15)
        reshape2 = reshape4.view(1, 2, 2, 15)
        o4 = self.final_4(reshape4.permute(0, 2, 1))
        o2 = self.final_2(reshape2.permute(0, 3, 2, 1))
        return o2, o4


class ReshapeReshape0Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(3, 60)
        self.final_2 = nn.Linear(2, 5)

    def forward(self, x):
        base = self.base(x)
        o4 = base.view(1, 4, 15)
        reshape2 = o4.view(1, 2, 2, 15)
        o2 = self.final_2(reshape2.permute(0, 3, 2, 1))
        return o2, o4


def get_swin_tiny_model():
    config = AutoConfig.from_pretrained("microsoft/swin-base-patch4-window7-224")
    return AutoModelForImageClassification.from_config(config)


def get_mobile_bert_big_model():
    config = AutoConfig.from_pretrained("google/mobilebert-uncased")
    return AutoModelForQuestionAnswering.from_config(config)


@dataclass
class GroupTestDesc:
    model_desc: IModelDesc
    ref_groups: Optional[List[PruningGroup]] = None

    def __str__(self) -> str:
        return self.model_desc.model_name


SYNTHETIC_DESCS = [
    GroupTestDesc(
        model_desc=GeneralModelDesc(model_builder=SelfAttention, input_sample_sizes=([384, 768])),
        ref_groups=[
            PruningGroup(
                block=PruningBlock(64, 0),
                producers={ProducerInfo(1), ProducerInfo(2), ProducerInfo(3)},
                consumers={ConsumerInfo(17)},
            ),
        ],
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(model_builder=ReshapeReshape1Block, input_sample_sizes=([1, 3])),
        ref_groups=[
            PruningGroup(
                block=PruningBlock(size=2, offset=0),
                producers={ProducerInfo(1)},
                consumers={ConsumerInfo(7), ConsumerInfo(5)},
            ),
        ],
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(model_builder=ReshapeReshape0Block, input_sample_sizes=([1, 3])), ref_groups=[]
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_builder=DiffNumBranchesOnJoining, input_sample_sizes=DiffNumBranchesOnJoining.INPUT_SAMPLE_SIZES
        ),
        ref_groups=[],
    ),
]

NLP_DESCS = [
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name="1_layer_BERT",
            input_info=[dict(sample_size=[1, 10], type="long")],
            model_builder=partial(AutoModelForQuestionAnswering.from_config, BertConfig(num_hidden_layers=1)),
        ),
        ref_groups=[
            PruningGroup(
                block=PruningBlock(size=64, offset=0),
                producers={ProducerInfo(11), ProducerInfo(14), ProducerInfo(10)},
                consumers={ConsumerInfo(29)},
            ),
            PruningGroup(block=PruningBlock(), producers={ProducerInfo(33)}, consumers={ConsumerInfo(35)}),
        ],
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name="larger_BERT",
            input_info=[dict(sample_size=[1, 128], type="long")],
            model_builder=partial(
                AutoModelForSequenceClassification.from_config,
                BertConfig(
                    hidden_size=4,
                    intermediate_size=3,
                    max_position_embeddings=128,
                    num_attention_heads=2,
                    num_hidden_layers=1,
                    vocab_size=10,
                    num_labels=2,
                    mhsa_qkv_bias=True,
                    mhsa_o_bias=True,
                    ffn_bias=True,
                ),
            ),
        ),
        ref_groups=[
            PruningGroup(
                block=PruningBlock(size=2, offset=0),
                producers={ProducerInfo(10), ProducerInfo(11), ProducerInfo(14)},
                consumers={ConsumerInfo(29)},
            ),
            PruningGroup(block=PruningBlock(), producers={ProducerInfo(33)}, consumers={ConsumerInfo(35)}),
            PruningGroup(block=PruningBlock(), producers={ProducerInfo(40)}, consumers={ConsumerInfo(43)}),
        ],
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name="RoBERTa",
            input_info=[dict(sample_size=[1, 10], type="long")],
            model_builder=partial(AutoModelForQuestionAnswering.from_config, RobertaConfig(num_hidden_layers=1)),
        ),
        ref_groups=[
            PruningGroup(
                block=PruningBlock(size=64, offset=0),
                producers={ProducerInfo(19), ProducerInfo(22), ProducerInfo(18)},
                consumers={ConsumerInfo(37)},
            ),
            PruningGroup(block=PruningBlock(), producers={ProducerInfo(41)}, consumers={ConsumerInfo(43)}),
        ],
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name="DistilBERT",
            input_info=[dict(sample_size=[1, 4], type="long")] * 2,
            model_builder=partial(
                AutoModelForQuestionAnswering.from_config,
                DistilBertConfig(
                    vocab_size=4,
                    max_position_embeddings=4,
                    n_layers=1,
                    n_heads=2,
                    dim=4,
                    hidden_dim=4 * 4,
                ),
            ),
        ),
        ref_groups=[
            PruningGroup(
                block=PruningBlock(size=2, offset=0),
                producers={ProducerInfo(7), ProducerInfo(10), ProducerInfo(13)},
                consumers={ConsumerInfo(29)},
            ),
            PruningGroup(block=PruningBlock(), producers={ProducerInfo(32)}, consumers={ConsumerInfo(34)}),
        ],
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name="MobileBERT",
            input_info=[dict(sample_size=[1, 128], type="long")],
            model_builder=partial(
                AutoModelForSequenceClassification.from_config,
                MobileBertConfig(
                    hidden_size=4,
                    intermediate_size=3,
                    max_position_embeddings=128,
                    num_attention_heads=2,
                    num_hidden_layers=1,
                    vocab_size=10,
                    num_labels=2,
                    mhsa_qkv_bias=True,
                    mhsa_o_bias=True,
                    ffn_bias=True,
                ),
            ),
        ),
        ref_groups=[
            PruningGroup(
                block=PruningBlock(),
                producers={
                    ProducerInfo(17),
                    ProducerInfo(42),
                    ProducerInfo(48),
                    ProducerInfo(54),
                    ProducerInfo(60),
                    ProducerInfo(66),
                },
                consumers={ConsumerInfo(64), ConsumerInfo(46), ConsumerInfo(58), ConsumerInfo(70), ConsumerInfo(52)},
            ),
            PruningGroup(
                block=PruningBlock(size=64, offset=0),
                producers={ProducerInfo(23), ProducerInfo(24), ProducerInfo(25)},
                consumers={ConsumerInfo(42)},
            ),
            PruningGroup(
                block=PruningBlock(), producers={ProducerInfo(20)}, consumers={ConsumerInfo(23), ConsumerInfo(24)}
            ),
            PruningGroup(block=PruningBlock(), producers={ProducerInfo(46)}, consumers={ConsumerInfo(48)}),
            PruningGroup(block=PruningBlock(), producers={ProducerInfo(52)}, consumers={ConsumerInfo(54)}),
            PruningGroup(block=PruningBlock(), producers={ProducerInfo(58)}, consumers={ConsumerInfo(60)}),
            PruningGroup(block=PruningBlock(), producers={ProducerInfo(64)}, consumers={ConsumerInfo(66)}),
            PruningGroup(block=PruningBlock(), producers={ProducerInfo(76)}, consumers={ConsumerInfo(79)}),
        ],
    ),
]

CV_DESCS = [
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name="Swin",
            input_info=dict(sample_size=[1, 3, 224, 224]),
            model_builder=partial(
                AutoModelForImageClassification.from_config,
                SwinConfig(
                    depths=[1],
                    num_heads=[2],
                    image_size=224,
                    patch_size=4,
                    num_channels=3,
                    embed_dim=4,
                ),
            ),
        ),
        ref_groups=[
            PruningGroup(
                block=PruningBlock(size=2, offset=0),
                producers={ProducerInfo(15), ProducerInfo(18), ProducerInfo(14)},
                consumers={ConsumerInfo(33)},
            ),
            PruningGroup(block=PruningBlock(), producers={ProducerInfo(43)}, consumers={ConsumerInfo(45)}),
        ],
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name="ViT_no_heads",
            input_info=dict(sample_size=[1, 1, 4, 4]),
            model_builder=partial(
                AutoModelForImageClassification.from_config,
                ViTConfig(
                    hidden_size=2,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=2,
                    image_size=4,
                    patch_size=2,
                    num_channels=1,
                ),
            ),
        ),
        ref_groups=[
            PruningGroup(
                block=PruningBlock(),
                producers={ProducerInfo(12), ProducerInfo(9), ProducerInfo(8)},
                consumers={ConsumerInfo(26)},
            ),
            PruningGroup(block=PruningBlock(), producers={ProducerInfo(30)}, consumers={ConsumerInfo(32)}),
        ],
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name="ViT",
            input_info=dict(sample_size=[1, 1, 4, 4]),
            model_builder=partial(
                AutoModelForImageClassification.from_config,
                ViTConfig(
                    hidden_size=4,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=2,
                    image_size=4,
                    patch_size=2,
                    num_channels=1,
                ),
            ),
        ),
        ref_groups=[
            PruningGroup(
                block=PruningBlock(size=2, offset=0),
                producers={ProducerInfo(12), ProducerInfo(9), ProducerInfo(8)},
                consumers={ConsumerInfo(26)},
            ),
            PruningGroup(block=PruningBlock(), producers={ProducerInfo(30)}, consumers={ConsumerInfo(32)}),
        ],
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name="CLIP",
            input_info=[dict(sample_size=[1, 3, 3, 3], type="float")] * 1,
            model_builder=partial(
                CLIPVisionModel,
                CLIPVisionConfig(
                    hidden_size=4,
                    intermediate_size=2,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_channels=3,
                    image_size=3,
                    patch_size=3,
                ),
            ),
        ),
        ref_groups=[
            PruningGroup(
                block=PruningBlock(size=2, offset=0),
                producers={ProducerInfo(10), ProducerInfo(16), ProducerInfo(12)},
                consumers={ConsumerInfo(34)},
            ),
            PruningGroup(block=PruningBlock(), producers={ProducerInfo(37)}, consumers={ConsumerInfo(41)}),
        ],
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name="Swin_MS",
            input_info=dict(sample_size=[1, 4 * 4, 8]),
            model_builder=partial(SwinTransformerBlock, dim=8, input_resolution=[4, 4], num_heads=2),
        ),
        ref_groups=[
            PruningGroup(
                block=PruningBlock(size=4, offset=8), producers={ProducerInfo(8, 0)}, consumers={ConsumerInfo(23, 1)}
            ),
            PruningGroup(
                block=PruningBlock(size=1, offset=0), producers={ProducerInfo(33, 0)}, consumers={ConsumerInfo(36, 1)}
            ),
        ],
    ),
]


AUDIO_DESCS = [
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name="Wave2Vec_2.0",
            input_info=dict(sample_size=[1, 400]),
            model_builder=partial(
                AutoModelForAudioClassification.from_config,
                Wav2Vec2Config(
                    vocab_size=2,
                    hidden_size=48,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=4,
                    conv_dim=(2, 2, 2, 2, 2, 2, 2),
                ),
            ),
        ),
        ref_groups=[
            PruningGroup(
                block=PruningBlock(size=24, offset=0),
                producers={ProducerInfo(31), ProducerInfo(29), ProducerInfo(35)},
                consumers={ConsumerInfo(53)},
            ),
            PruningGroup(block=PruningBlock(), producers={ProducerInfo(57)}, consumers={ConsumerInfo(60)}),
        ],
    ),
]


TEST_DESCS = [*SYNTHETIC_DESCS, *NLP_DESCS, *CV_DESCS, *AUDIO_DESCS]


@pytest.mark.parametrize("desc", TEST_DESCS, ids=map(str, TEST_DESCS))
def test_groups(desc: GroupTestDesc, mocker, tmp_path):
    model_desc = desc.model_desc
    model = model_desc.get_model()
    config = NNCFConfig({"input_info": model_desc.create_input_info()})
    nncf_network = create_nncf_network(model, config)
    pruning_producing_types = ["linear"]
    get_graph_spy = mocker.spy(BlockHierarchy, "_get_graph_for_visualization")
    not_filtered_groups = get_pruning_groups(
        nncf_network.nncf.get_graph(), PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES, pruning_producing_types, tmp_path
    )

    nx_graph = get_graph_spy.spy_return
    path_to_dot = get_full_path_to_the_graph(f"{str(desc)}.dot", "pruning_groups")
    compare_nx_graph_with_reference(nx_graph, path_to_dot, sort_dot_graph=False)

    filtered_groups = select_largest_groups(not_filtered_groups)
    assert filtered_groups == desc.ref_groups


BIG_MODEL_DESCS = [
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name="MobileBERT_big",
            input_info=[dict(sample_size=[1, 128], type="long")] * 4,
            model_builder=get_mobile_bert_big_model,
        ),
    ),
    GroupTestDesc(
        model_desc=GeneralModelDesc(
            model_name="Swin_big",
            input_info=dict(sample_size=[1, 3, 224, 224]),
            model_builder=get_swin_tiny_model,
        )
    ),
]


@pytest.mark.parametrize("desc", BIG_MODEL_DESCS, ids=map(str, BIG_MODEL_DESCS))
def test_all_groups_valid(desc: GroupTestDesc):
    model_desc = desc.model_desc
    model = model_desc.get_model()
    config = NNCFConfig({"input_info": model_desc.create_input_info()})
    nncf_network = create_nncf_network(model, config)
    pruning_producing_types = ["linear"]
    all_groups = get_pruning_groups(
        nncf_network.nncf.get_graph(), PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES, pruning_producing_types
    )
    for group in all_groups:
        assert group.consumers
        assert group.producers
