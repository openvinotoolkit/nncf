# Copyright (c) 2023 Intel Corporation
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
from typing import Dict, List, Set

import pytest

from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.quantization.quantizer_setup import MultiConfigQuantizationPoint
from nncf.common.quantization.quantizer_setup import MultiConfigQuantizerSetup
from nncf.common.quantization.quantizer_setup import QuantizationPointId
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import UnifiedScaleType

QCONFIG_PER_CHANNEL = QuantizerConfig(per_channel=True)
QCONFIG_PER_TENSOR = QuantizerConfig(per_channel=False)


@pytest.fixture()
def multiconf_quant_setup_for_unified_scale_testing() -> MultiConfigQuantizerSetup:
    setup = MultiConfigQuantizerSetup()
    setup.quantization_points = {
        0: MultiConfigQuantizationPoint(
            ActivationQuantizationInsertionPoint(target_node_name="A"), [QCONFIG_PER_CHANNEL, QCONFIG_PER_TENSOR], []
        ),
        1: MultiConfigQuantizationPoint(
            ActivationQuantizationInsertionPoint(target_node_name="B"), [QCONFIG_PER_CHANNEL, QCONFIG_PER_TENSOR], []
        ),
        2: MultiConfigQuantizationPoint(
            ActivationQuantizationInsertionPoint(target_node_name="C"), [QCONFIG_PER_CHANNEL, QCONFIG_PER_TENSOR], []
        ),
        3: MultiConfigQuantizationPoint(
            ActivationQuantizationInsertionPoint(target_node_name="D"), [QCONFIG_PER_CHANNEL, QCONFIG_PER_TENSOR], []
        ),
    }
    return setup


@dataclass
class UnifiedScaleAssignmentTestStruct:
    unified_groups: List[Set[QuantizationPointId]]
    unification_types: Dict[QuantizationPointId, UnifiedScaleType]
    qconfig_choices: Dict[QuantizationPointId, QuantizerConfig]
    ref_unified_groups_after_qconfig_selection: List[Set[QuantizationPointId]]


@pytest.mark.parametrize(
    "test_struct",
    [
        UnifiedScaleAssignmentTestStruct(
            unified_groups=[{0, 1}],
            unification_types={0: UnifiedScaleType.UNIFY_ALWAYS, 1: UnifiedScaleType.UNIFY_ALWAYS},
            qconfig_choices={
                0: QCONFIG_PER_TENSOR,
                1: QCONFIG_PER_TENSOR,
                2: QCONFIG_PER_TENSOR,
                3: QCONFIG_PER_TENSOR,
            },
            ref_unified_groups_after_qconfig_selection=[{0, 1}],
        ),
        UnifiedScaleAssignmentTestStruct(
            unified_groups=[{1, 2, 3}],
            unification_types={
                1: UnifiedScaleType.UNIFY_ALWAYS,
                2: UnifiedScaleType.UNIFY_ALWAYS,
                3: UnifiedScaleType.UNIFY_ALWAYS,
            },
            qconfig_choices={
                0: QCONFIG_PER_TENSOR,
                1: QCONFIG_PER_TENSOR,
                2: QCONFIG_PER_TENSOR,
                3: QCONFIG_PER_TENSOR,
            },
            ref_unified_groups_after_qconfig_selection=[{1, 2, 3}],
        ),
        UnifiedScaleAssignmentTestStruct(
            unified_groups=[{0, 1}],
            unification_types={0: UnifiedScaleType.UNIFY_ALWAYS, 1: UnifiedScaleType.UNIFY_ALWAYS},
            qconfig_choices={
                0: QCONFIG_PER_CHANNEL,
                1: QCONFIG_PER_CHANNEL,
                2: QCONFIG_PER_CHANNEL,
                3: QCONFIG_PER_TENSOR,
            },
            ref_unified_groups_after_qconfig_selection=[{0, 1}],
        ),
        UnifiedScaleAssignmentTestStruct(
            unified_groups=[{0, 1}],
            unification_types={0: UnifiedScaleType.UNIFY_ONLY_PER_TENSOR, 1: UnifiedScaleType.UNIFY_ALWAYS},
            qconfig_choices={
                0: QCONFIG_PER_CHANNEL,
                1: QCONFIG_PER_CHANNEL,
                2: QCONFIG_PER_CHANNEL,
                3: QCONFIG_PER_TENSOR,
            },
            ref_unified_groups_after_qconfig_selection=[],
        ),
        UnifiedScaleAssignmentTestStruct(
            unified_groups=[{1, 2, 3}],
            unification_types={
                1: UnifiedScaleType.UNIFY_ONLY_PER_TENSOR,
                2: UnifiedScaleType.UNIFY_ONLY_PER_TENSOR,
                3: UnifiedScaleType.UNIFY_ONLY_PER_TENSOR,
            },
            qconfig_choices={
                0: QCONFIG_PER_TENSOR,
                1: QCONFIG_PER_CHANNEL,
                2: QCONFIG_PER_TENSOR,
                3: QCONFIG_PER_CHANNEL,
            },
            ref_unified_groups_after_qconfig_selection=[],
        ),
        UnifiedScaleAssignmentTestStruct(
            unified_groups=[{1, 2, 3}],
            unification_types={
                1: UnifiedScaleType.UNIFY_ONLY_PER_TENSOR,
                2: UnifiedScaleType.UNIFY_ONLY_PER_TENSOR,
                3: UnifiedScaleType.UNIFY_ONLY_PER_TENSOR,
            },
            qconfig_choices={
                0: QCONFIG_PER_TENSOR,
                1: QCONFIG_PER_TENSOR,
                2: QCONFIG_PER_CHANNEL,
                3: QCONFIG_PER_TENSOR,
            },
            ref_unified_groups_after_qconfig_selection=[{1, 3}],
        ),
        UnifiedScaleAssignmentTestStruct(
            unified_groups=[{0, 2}, {1, 3}],
            unification_types={
                0: UnifiedScaleType.UNIFY_ALWAYS,
                1: UnifiedScaleType.UNIFY_ALWAYS,
                2: UnifiedScaleType.UNIFY_ALWAYS,
                3: UnifiedScaleType.UNIFY_ALWAYS,
            },
            qconfig_choices={
                0: QCONFIG_PER_TENSOR,
                1: QCONFIG_PER_CHANNEL,
                2: QCONFIG_PER_TENSOR,
                3: QCONFIG_PER_CHANNEL,
            },
            ref_unified_groups_after_qconfig_selection=[{0, 2}, {1, 3}],
        ),
        UnifiedScaleAssignmentTestStruct(
            unified_groups=[{0, 1, 2, 3}],
            unification_types={
                0: UnifiedScaleType.UNIFY_ALWAYS,
                1: UnifiedScaleType.UNIFY_ALWAYS,
                2: UnifiedScaleType.UNIFY_ALWAYS,
                3: UnifiedScaleType.UNIFY_ALWAYS,
            },
            qconfig_choices={
                0: QCONFIG_PER_TENSOR,
                1: QCONFIG_PER_CHANNEL,
                2: QCONFIG_PER_TENSOR,
                3: QCONFIG_PER_CHANNEL,
            },
            ref_unified_groups_after_qconfig_selection=[{0, 2}, {1, 3}],
        ),
    ],
)
def test_unified_scale_assignment_based_on_qconfig_selection(
    test_struct: UnifiedScaleAssignmentTestStruct,
    multiconf_quant_setup_for_unified_scale_testing: MultiConfigQuantizerSetup,
):
    multi_setup = multiconf_quant_setup_for_unified_scale_testing
    for us_group in test_struct.unified_groups:
        us_group_list = list(us_group)
        multi_setup.register_unified_scale_group_with_types(
            us_group_list, [test_struct.unification_types[qp_id] for qp_id in us_group_list]
        )

    single_setup = multi_setup.select_qconfigs(test_struct.qconfig_choices)

    assert list(single_setup.unified_scale_groups.values()) == test_struct.ref_unified_groups_after_qconfig_selection
