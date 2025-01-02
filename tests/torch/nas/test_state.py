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
import logging
from copy import deepcopy
from functools import partial

import pytest
import torch

from nncf.common.logging import nncf_logger
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import SEHBuilderStateNames
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_depth import EDBuilderStateNames
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_kernel import EKBuilderStateNames
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_width import EWBuilderStateNames
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.torch.model_creation import create_nncf_network
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import get_empty_config
from tests.torch.nas.creators import build_elastic_model_from_handler
from tests.torch.nas.descriptors import ElasticityDesc
from tests.torch.nas.helpers import do_conv2d
from tests.torch.nas.helpers import move_model_to_cuda_if_available
from tests.torch.nas.test_elastic_depth import BASIC_ELASTIC_DEPTH_PARAMS
from tests.torch.nas.test_elastic_depth import BasicTestSuperNet
from tests.torch.nas.test_elastic_depth import DepthBasicConvTestModel
from tests.torch.nas.test_elastic_kernel import BASIC_ELASTIC_KERNEL_PARAMS
from tests.torch.nas.test_elastic_width import BASIC_ELASTIC_WIDTH_PARAMS
from tests.torch.nas.test_elastic_width import TwoConvAddConvTestModel
from tests.torch.nas.test_elastic_width import TwoSequentialConvBNTestModel


@pytest.fixture
def _nncf_caplog(caplog):
    nncf_logger.propagate = True
    yield caplog
    nncf_logger.propagate = False


def ref_width_output_fn(model, x):
    return model.get_minimal_subnet_output_without_reorg(x)


COMMON_WIDTH_STATE_DESCS = [
    ElasticityDesc(
        ElasticityDim.WIDTH,
        model_cls=TwoConvAddConvTestModel,
        params=BASIC_ELASTIC_WIDTH_PARAMS,
        ref_state={
            "elasticity_params": BASIC_ELASTIC_WIDTH_PARAMS,
            "grouped_node_names_to_prune": [
                [
                    "TwoConvAddConvTestModel/NNCFConv2d[conv1]/conv2d_0",
                    "TwoConvAddConvTestModel/NNCFConv2d[conv2]/conv2d_0",
                ]
            ],
        },
        ref_output_fn=ref_width_output_fn,
    ),
    ElasticityDesc(
        ElasticityDim.WIDTH,
        model_cls=TwoSequentialConvBNTestModel,
        params=BASIC_ELASTIC_WIDTH_PARAMS,
        ref_state={
            "elasticity_params": BASIC_ELASTIC_WIDTH_PARAMS,
            "grouped_node_names_to_prune": [
                ["TwoSequentialConvBNTestModel/Sequential[all_layers]/NNCFConv2d[0]/conv2d_0"],
                ["TwoSequentialConvBNTestModel/Sequential[all_layers]/NNCFConv2d[3]/conv2d_0"],
            ],
        },
        ref_output_fn=ref_width_output_fn,
    ),
]


def ref_kernel_output_fn(model, x):
    conv = model.conv
    ref_padding = 1
    ref_weights = conv.weight[:, :, 1:4, 1:4]
    return do_conv2d(conv, x, weight=ref_weights, padding=ref_padding)


COMMON_KERNEL_DESC = ElasticityDesc(
    ElasticityDim.KERNEL,
    model_cls=partial(BasicConvTestModel, 1, out_channels=1, kernel_size=5, padding=2),
    params=BASIC_ELASTIC_KERNEL_PARAMS,
    ref_output_fn=ref_kernel_output_fn,
    ref_state={
        SEHBuilderStateNames.ELASTICITY_PARAMS: BASIC_ELASTIC_KERNEL_PARAMS,
        EKBuilderStateNames.NODE_NAMES_TO_MAKE_ELASTIC: ["BasicConvTestModel/NNCFConv2d[conv]/conv2d_0"],
    },
    input_size=[1, 1, 5, 5],
)

COMMON_DEPTH_SUPERNET_DESC = ElasticityDesc(
    ElasticityDim.DEPTH,
    model_cls=BasicTestSuperNet,
    params={
        "min_block_size": 2,
        "hw_fused_ops": True,
    },
    ref_state={
        "elasticity_params": {"hw_fused_ops": True, "max_block_size": 50, "min_block_size": 2, "skipped_blocks": None},
        EDBuilderStateNames.SKIPPED_BLOCKS: [],
        EDBuilderStateNames.SKIPPED_BLOCKS_DEPENDENCIES: {},
    },
    ref_search_space=[[]],
)


def ref_depth_output_fn(model: DepthBasicConvTestModel, x):
    skipped_layers_before = model.get_skipped_layers()
    model.set_skipped_layers(["conv1"])
    result = model(x)
    model.set_skipped_layers(skipped_layers_before)
    return result


COMMON_DEPTH_BASIC_DESC = ElasticityDesc(
    ElasticityDim.DEPTH,
    model_cls=DepthBasicConvTestModel,
    params=BASIC_ELASTIC_DEPTH_PARAMS,
    ref_output_fn=ref_depth_output_fn,
    ref_search_space=[[0], []],
    ref_state={
        "elasticity_params": {
            "hw_fused_ops": True,
            "max_block_size": 50,
            "min_block_size": 5,
            "skipped_blocks": [
                [
                    "DepthBasicConvTestModel/Sequential[branch_with_blocks]/NNCFConv2d[conv0]/conv2d_0",
                    "DepthBasicConvTestModel/Sequential[branch_with_blocks]/NNCFConv2d[conv1]/conv2d_0",
                ]
            ],
        },
        EDBuilderStateNames.SKIPPED_BLOCKS: BASIC_ELASTIC_DEPTH_PARAMS["skipped_blocks_state"],
        EDBuilderStateNames.SKIPPED_BLOCKS_DEPENDENCIES: BASIC_ELASTIC_DEPTH_PARAMS["skipped_blocks_dependencies"],
    },
)

LIST_STATE_AFTER_BUILD_DESCS = [*COMMON_WIDTH_STATE_DESCS, COMMON_DEPTH_SUPERNET_DESC, COMMON_KERNEL_DESC]


@pytest.mark.parametrize("desc", LIST_STATE_AFTER_BUILD_DESCS, ids=map(str, LIST_STATE_AFTER_BUILD_DESCS))
def test_can_get_builder_state_after_build(desc):
    _, builder = desc.build_handler()
    actual_state = builder.get_state()
    assert actual_state == desc.ref_state


ELASTIC_WIDTH_PARAMS_BB = {"filter_importance": "L2", **BASIC_ELASTIC_WIDTH_PARAMS}
LIST_STATE_BEFORE_BUILD_DESCS = [
    ElasticityDesc(
        ElasticityDim.WIDTH,
        params=ELASTIC_WIDTH_PARAMS_BB,
        ref_state={
            SEHBuilderStateNames.ELASTICITY_PARAMS: ELASTIC_WIDTH_PARAMS_BB,
            EWBuilderStateNames.GROUPED_NODE_NAMES_TO_PRUNE: [],
        },
    ),
    ElasticityDesc(
        ElasticityDim.KERNEL,
        params=BASIC_ELASTIC_KERNEL_PARAMS,
        ref_state={
            SEHBuilderStateNames.ELASTICITY_PARAMS: BASIC_ELASTIC_KERNEL_PARAMS,
            EKBuilderStateNames.NODE_NAMES_TO_MAKE_ELASTIC: [],
        },
    ),
    COMMON_DEPTH_BASIC_DESC,
]


@pytest.mark.parametrize("desc", LIST_STATE_BEFORE_BUILD_DESCS, ids=map(str, LIST_STATE_BEFORE_BUILD_DESCS))
class TestBeforeBuild:
    def test_can_get_builder_state_before_build(self, desc: ElasticityDesc):
        builder = desc.create_builder()
        actual_state = builder.get_state()
        assert actual_state == desc.ref_state

    def test_output_warning_when_state_overrides_params(self, desc: ElasticityDesc, _nncf_caplog):
        old_builder = desc.create_builder_with_config({})
        old_state = old_builder.get_state()

        new_params = desc.params
        new_builder = desc.create_builder_with_config(new_params)
        new_builder.load_state(old_state)

        record = next(iter(_nncf_caplog.records))
        assert record.levelno == logging.WARNING

    def test_no_warning_when_state_and_params_are_the_same(self, desc: ElasticityDesc, _nncf_caplog):
        old_builder = desc.create_builder()
        old_state = old_builder.get_state()

        new_params = desc.params.copy()
        new_builder = desc.create_builder_with_config(new_params)
        new_builder.load_state(old_state)

        assert not _nncf_caplog.records


LIST_LOAD_STATE_DESCS = [COMMON_DEPTH_BASIC_DESC, *COMMON_WIDTH_STATE_DESCS, COMMON_KERNEL_DESC]


@pytest.mark.parametrize("desc", LIST_LOAD_STATE_DESCS, ids=map(str, LIST_LOAD_STATE_DESCS))
def test_can_load_handler_state(desc: ElasticityDesc):
    original_model = desc.model_cls()
    move_model_to_cuda_if_available(original_model)
    original_model.eval()
    model = deepcopy(original_model)
    device = next(iter(model.parameters())).device
    dummy_input = torch.ones(model.INPUT_SIZE).to(device)

    input_size = desc.input_size
    if not input_size:
        input_size = model.INPUT_SIZE
    config = get_empty_config(input_sample_sizes=input_size)
    old_nncf_network = create_nncf_network(model, config)
    old_builder = desc.create_builder()
    old_handler = old_builder.build(old_nncf_network)
    elastic_model = build_elastic_model_from_handler(old_nncf_network, old_handler)
    old_handler.activate_minimum_subnet()
    old_output = elastic_model(dummy_input)
    ref_output = desc.ref_output_fn(original_model, dummy_input)
    assert torch.allclose(old_output, ref_output)

    new_nncf_network = create_nncf_network(deepcopy(original_model), config)
    builder_state = old_builder.get_state()
    # no need in config to restore builder state
    new_builder = desc.create_builder_with_config({})

    new_builder.load_state(builder_state)
    new_handler = new_builder.build(new_nncf_network)
    elastic_model = build_elastic_model_from_handler(new_nncf_network, new_handler)
    new_handler.activate_minimum_subnet()
    new_output = elastic_model(dummy_input)
    assert torch.allclose(old_output, new_output)
