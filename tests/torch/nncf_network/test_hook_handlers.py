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

from typing import Callable, List, Tuple

import pytest
import torch

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.hook_handle import HookHandle
from nncf.torch.dynamic_graph.io_handling import ExampleInputInfo
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.nncf_network import PTInsertionPoint
from tests.torch.helpers import HookChecker
from tests.torch.nncf_network.helpers import SimplestModel


@pytest.mark.parametrize(
    "target_type, target_node_name, input_port_id",
    [
        (TargetType.OPERATOR_PRE_HOOK, "/nncf_model_output_0", 0),
        (TargetType.OPERATOR_POST_HOOK, "/nncf_model_input_0", 0),
        (TargetType.PRE_LAYER_OPERATION, "SimplestModel/NNCFConv2d[conv]/conv2d_0", 0),
        (TargetType.POST_LAYER_OPERATION, "SimplestModel/NNCFConv2d[conv]/conv2d_0", 0),
    ],
)
class TestHookHandles:
    class HookTest(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros((1,)))

        def forward(self, x):
            return x + self._p

    @staticmethod
    def _prepare_hook_handles_test(
        target_type: TargetType, target_node_name: str, input_port_id: int
    ) -> Tuple[NNCFNetwork, PTInsertionPoint, Callable[[List[HookHandle]], None]]:
        model = SimplestModel()
        example_input = torch.ones(SimplestModel.INPUT_SIZE)
        input_info = ExampleInputInfo.from_example_input(example_input)
        nncf_model = NNCFNetwork(model, input_info)

        node_name_vs_address = nncf_model.nncf.get_node_to_op_address_mapping()
        ip = PTInsertionPoint(target_type, node_name_vs_address[target_node_name], input_port_id=input_port_id)

        checker = HookChecker(nncf_model, "conv")

        def _check(ref_hooks_):
            checker.clear()
            checker.add_ref(ref_hooks_, target_type, target_node_name, input_port_id)
            checker.check_with_reference()

        return nncf_model, ip, _check

    def test_temporary_insert_at_point_by_hook_group_name(
        self, target_type: TargetType, target_node_name: str, input_port_id: int
    ):
        nncf_model, ip, _check = self._prepare_hook_handles_test(target_type, target_node_name, input_port_id)
        permanent_hook = self.HookTest()
        TEMPORARY_HOOK_GROUP_NAME = "tmp"
        # Make temporary hook a ref to the permanent hook
        # to check tmp hooks are not removed by their id()
        temporary_hook = permanent_hook
        nncf_model.nncf.insert_at_point(ip, permanent_hook)
        ref_hooks = [permanent_hook]
        _check(ref_hooks)

        for _ in range(2):
            temporary_hook = self.HookTest()
            nncf_model.nncf.insert_at_point(ip, temporary_hook, TEMPORARY_HOOK_GROUP_NAME)
            ref_hooks.append(temporary_hook)
            _check(ref_hooks)

            nncf_model.nncf.insert_at_point(ip, permanent_hook)
            ref_hooks.append(permanent_hook)
            _check(ref_hooks)

            nncf_model.nncf.remove_hooks_group(TEMPORARY_HOOK_GROUP_NAME)
            del ref_hooks[-2]
            _check(ref_hooks)
            assert not nncf_model.nncf._groups_vs_hooks_handlers[TEMPORARY_HOOK_GROUP_NAME]

    def test_insert_at_point_hook_handles(self, target_type: TargetType, target_node_name: str, input_port_id: int):
        nncf_model, ip, _check = self._prepare_hook_handles_test(target_type, target_node_name, input_port_id)
        permanent_hook = self.HookTest()
        # Make temporary hook a ref to the permanent hook
        # to check tmp hooks are not removed by their id()
        temporary_hook = permanent_hook
        tmp_hh = []
        nncf_model.nncf.insert_at_point(ip, permanent_hook)

        ref_hooks = [permanent_hook]
        _check(ref_hooks)

        for _ in range(2):
            temporary_hook = self.HookTest()
            tmp_hh.append(nncf_model.nncf.insert_at_point(ip, temporary_hook))
            ref_hooks.append(temporary_hook)
            _check(ref_hooks)

            nncf_model.nncf.insert_at_point(ip, permanent_hook)
            ref_hooks.append(permanent_hook)
            _check(ref_hooks)

            for hh in tmp_hh:
                hh.remove()

            del ref_hooks[-2]
            _check(ref_hooks)
