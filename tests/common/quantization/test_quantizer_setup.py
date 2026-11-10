# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.quantizer_setup import WeightQuantizationInsertionPoint
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.stateful_classes_registry import CommonStatefulClassesRegistry


class TestSingleConfigQuantizationPointRegistry:
    def test_is_registered_in_common_stateful_classes_registry(self):
        registered_cls = CommonStatefulClassesRegistry.get_registered_class(SingleConfigQuantizationPoint.__name__)
        assert registered_cls is SingleConfigQuantizationPoint

    @pytest.mark.parametrize(
        "insertion_point,node_name",
        [
            (WeightQuantizationInsertionPoint("node_A"), "node_A"),
            (ActivationQuantizationInsertionPoint("node_B", input_port_id=0), "node_B"),
            (ActivationQuantizationInsertionPoint("node_C", input_port_id=None), "node_C"),
        ],
    )
    def test_get_state_and_from_state_via_registry(self, insertion_point, node_name):
        qconfig = QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False)
        original = SingleConfigQuantizationPoint(insertion_point, qconfig, [node_name])
        qp_cls = CommonStatefulClassesRegistry.get_registered_class(SingleConfigQuantizationPoint.__name__)
        restored = qp_cls.from_state(original.get_state())
        assert restored == original


class TestSingleConfigQuantizerSetupFromState:
    def _make_setup(self) -> SingleConfigQuantizerSetup:
        setup = SingleConfigQuantizerSetup()

        weight_ip = WeightQuantizationInsertionPoint("conv/weight")
        act_ip = ActivationQuantizationInsertionPoint("relu/output", input_port_id=0)
        qconfig_sym = QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False)
        qconfig_asym = QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=True)

        qp_weight = SingleConfigQuantizationPoint(weight_ip, qconfig_sym, ["conv/weight"])
        qp_act = SingleConfigQuantizationPoint(act_ip, qconfig_asym, ["relu/output"])

        setup.add_independent_quantization_point(qp_weight)
        setup.add_independent_quantization_point(qp_act)
        return setup

    def test_from_state_roundtrip(self):
        original = self._make_setup()
        state = original.get_state()
        restored = SingleConfigQuantizerSetup.from_state(state)

        assert len(restored.quantization_points) == len(original.quantization_points)
        for qp_id, orig_qp in original.quantization_points.items():
            assert qp_id in restored.quantization_points
            assert restored.quantization_points[qp_id] == orig_qp

    def test_from_state_restores_unified_scale_groups(self):
        setup = self._make_setup()
        qp_ids = list(setup.quantization_points.keys())
        setup.register_unified_scale_group(qp_ids)

        state = setup.get_state()
        restored = SingleConfigQuantizerSetup.from_state(state)

        assert len(restored.unified_scale_groups) == 1
        restored_group = next(iter(restored.unified_scale_groups.values()))
        assert restored_group == set(qp_ids)
