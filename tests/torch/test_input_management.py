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

import inspect
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pytest
import torch

from nncf.torch.dynamic_graph.io_handling import ExampleInputInfo
from nncf.torch.dynamic_graph.io_handling import FillerInputElement
from nncf.torch.dynamic_graph.io_handling import FillerInputInfo
from nncf.torch.dynamic_graph.io_handling import InputInfoWrapManager
from nncf.torch.dynamic_graph.io_handling import ModelInputInfo
from tests.torch.helpers import MockModel
from tests.torch.helpers import ModelWithReloadedForward
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.test_compressed_graph import get_basic_quantization_config

TENSOR_1 = torch.ones([1]) * (-1)
TENSOR_2 = torch.ones([1]) * (-2)
TENSOR_3 = torch.ones([1]) * (-3)
TENSOR_4 = torch.ones([1]) * (-4)
TENSOR_DEFAULT = torch.ones([1]) * (-5)
TENSOR_FROM_INPUT_INFO_1 = torch.ones([1])
TENSOR_FROM_INPUT_INFO_2 = torch.ones([2])
TENSOR_FROM_INPUT_INFO_3 = torch.ones([3])


def forward(arg1=None, arg2=None, arg3=None, arg4=None, arg5=TENSOR_DEFAULT):
    pass


@dataclass
class InputWrappingTestStruct:
    input_info: ModelInputInfo
    model_args: Tuple
    model_kwargs: Dict
    ref_wrapping_sequence: List[torch.Tensor]
    case_id: str

    def get_case_id(self) -> str:
        return self.case_id + "-" + "filler" if isinstance(self.input_info, FillerInputInfo) else "exact"


INPUT_WRAPPING_TEST_CASES = [
    InputWrappingTestStruct(
        input_info=FillerInputInfo([FillerInputElement([1])]),
        model_args=(TENSOR_1,),
        model_kwargs={},
        ref_wrapping_sequence=[TENSOR_1],
        case_id="single_arg",
    ),
    InputWrappingTestStruct(
        input_info=FillerInputInfo([FillerInputElement([1], keyword="arg2")]),
        model_args=(),
        model_kwargs={"arg2": TENSOR_2},
        ref_wrapping_sequence=[TENSOR_2],
        case_id="single_kwarg",
    ),
    InputWrappingTestStruct(
        input_info=FillerInputInfo(
            [
                FillerInputElement([1]),
                FillerInputElement([1]),
                FillerInputElement([1], keyword="arg3"),
                FillerInputElement([1], keyword="arg5"),
            ]
        ),
        model_args=(TENSOR_1, TENSOR_2),
        model_kwargs={"arg3": TENSOR_3, "arg5": TENSOR_4},
        ref_wrapping_sequence=[TENSOR_1, TENSOR_2, TENSOR_3, TENSOR_4],
        case_id="args_and_kwargs",
    ),
    # More args supplied than what is specified by input_infos - ignore the unspecified args
    InputWrappingTestStruct(
        input_info=FillerInputInfo(
            [FillerInputElement([1]), FillerInputElement([1], keyword="arg3"), FillerInputElement([1], keyword="arg5")]
        ),
        model_args=(TENSOR_1, TENSOR_2),
        model_kwargs={"arg3": TENSOR_3, "arg5": TENSOR_4},
        ref_wrapping_sequence=[TENSOR_1, TENSOR_3, TENSOR_4],
        case_id="more_args_than_specified",
    ),
    # More args and kwargs supplied than what is specified by input_infos - ignore the unspecified args and kwargs
    InputWrappingTestStruct(
        input_info=FillerInputInfo([FillerInputElement([1]), FillerInputElement([1], keyword="arg4")]),
        model_args=(TENSOR_1, TENSOR_2),
        model_kwargs={"arg4": TENSOR_3, "arg5": TENSOR_4},
        ref_wrapping_sequence=[TENSOR_1, TENSOR_3],
        case_id="more_args_and_kwargs_than_specified",
    ),
    # arg specified, but kwarg supplied
    InputWrappingTestStruct(
        input_info=FillerInputInfo([FillerInputElement([1])]),
        model_args=(),
        model_kwargs={"arg3": TENSOR_1},
        ref_wrapping_sequence=[TENSOR_FROM_INPUT_INFO_1],
        case_id="kwarg_instead_of_arg",
    ),
    InputWrappingTestStruct(
        input_info=FillerInputInfo([FillerInputElement([1], keyword="arg5")]),
        model_args=(),
        model_kwargs={"arg1": TENSOR_1},
        ref_wrapping_sequence=[TENSOR_DEFAULT],
        case_id="arg_as_kwarg",
    ),
    # kwarg specified, but missing in supplied kwargs
    InputWrappingTestStruct(
        input_info=FillerInputInfo([FillerInputElement([1]), FillerInputElement([2], keyword="arg3")]),
        model_args=(TENSOR_1, TENSOR_2),
        model_kwargs={"arg4": TENSOR_3, "arg5": TENSOR_4},
        ref_wrapping_sequence=[TENSOR_1, TENSOR_FROM_INPUT_INFO_2],
        case_id="missing_kwarg",
    ),
    # More args specified than supplied
    InputWrappingTestStruct(
        input_info=FillerInputInfo(
            [FillerInputElement([1]), FillerInputElement([2]), FillerInputElement([3], keyword="arg3")]
        ),
        model_args=(TENSOR_1,),
        model_kwargs={"arg3": TENSOR_2},
        ref_wrapping_sequence=[TENSOR_1, TENSOR_FROM_INPUT_INFO_2, TENSOR_2],
        case_id="less_args_supplied",
    ),
    # More kwargs specified than supplied
    InputWrappingTestStruct(
        input_info=FillerInputInfo(
            [FillerInputElement([1]), FillerInputElement([2], keyword="arg2"), FillerInputElement([3], keyword="arg3")]
        ),
        model_args=(TENSOR_1,),
        model_kwargs={"arg2": TENSOR_2},
        ref_wrapping_sequence=[TENSOR_1, TENSOR_2, TENSOR_FROM_INPUT_INFO_3],
        case_id="less_kwargs_supplied",
    ),
]


@pytest.fixture(
    params=INPUT_WRAPPING_TEST_CASES,
    name="inputs_test_struct",
    ids=[x.get_case_id() for x in INPUT_WRAPPING_TEST_CASES],
)
def inputs_test_struct_(request):
    return request.param


def test_input_wrapper_wrap_inputs(mocker, inputs_test_struct: InputWrappingTestStruct):
    input_info = inputs_test_struct.input_info
    model_args = inputs_test_struct.model_args
    model_kwargs = inputs_test_struct.model_kwargs
    ref_wrapping_sequence = inputs_test_struct.ref_wrapping_sequence
    stub_cpu_model = MockModel()

    mocker.patch("nncf.torch.dynamic_graph.io_handling.nncf_model_input")
    from nncf.torch.dynamic_graph.io_handling import nncf_model_input

    mgr = InputInfoWrapManager(input_info, inspect.signature(forward), stub_cpu_model)
    mgr.wrap_inputs(model_args, model_kwargs)
    test_wrapping_sequence = [cl[0][0] for cl in nncf_model_input.call_args_list]

    test_identical_to_ref = all(map(torch.equal, ref_wrapping_sequence, test_wrapping_sequence))

    assert test_identical_to_ref


class CatModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_parameter = torch.nn.Parameter(torch.ones([1]))

    def forward(self, x, y):
        return torch.cat([x, y])


def test_same_input_tensor_replication(mocker):
    config = get_basic_quantization_config(
        input_info=[
            {"sample_size": [1, 1]},
            {"sample_size": [1, 1]},
        ]
    )
    register_bn_adaptation_init_args(config)
    model = CatModel()
    model, _ = create_compressed_model_and_algo_for_test(model, config)

    test_tensor = torch.ones([1, 1])
    clone_spy = mocker.spy(test_tensor, "clone")
    cat_spy = mocker.spy(torch, "cat")
    _ = model(test_tensor, test_tensor)
    assert clone_spy.call_count == 1
    cat_arg = cat_spy.call_args[0][0]
    assert cat_arg[0] is not cat_arg[1]


@pytest.mark.parametrize("use_kwargs", (False, True))
def test_reloaded_forward_inputs_wrapping(use_kwargs, mocker):
    """
    Check for a model which could accept both tensor and not tensor for the same
    input parameter that a non tensor input after a tensor input does not cause errors.
    """
    model = ModelWithReloadedForward()
    example_input = torch.ones(ModelWithReloadedForward.INPUT_SHAPE)
    input_info = ExampleInputInfo.from_example_input(example_input)
    mgr = InputInfoWrapManager(input_info, inspect.signature(model.forward), model)
    if use_kwargs:
        model_args = ()
        model_kwargs = {"x": {"tensor": example_input}}
    else:
        model_args = ({"tensor": example_input},)
        model_kwargs = {}

    ref_model_args = ({"tensor": example_input},)
    ref_model_kwargs = {}

    mocker.patch("nncf.torch.dynamic_graph.io_handling.nncf_model_input")
    from nncf.torch.dynamic_graph.io_handling import nncf_model_input

    actual_args, actual_kwargs = mgr.wrap_inputs(model_args, model_kwargs)
    nncf_model_input.assert_not_called()
    assert actual_args == ref_model_args
    assert actual_kwargs == ref_model_kwargs
