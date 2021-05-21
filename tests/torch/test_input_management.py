import inspect

import pytest
import torch

from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo
from nncf.torch.dynamic_graph.io_handling import InputInfoWrapManager
from tests.torch.helpers import MockModel
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


class InputWrappingTestStruct:
    def __init__(self, input_infos, model_args, model_kwargs, ref_wrapping_sequence):
        self.input_infos = input_infos
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.ref_wrapping_sequence = ref_wrapping_sequence


INPUT_WRAPPING_TEST_CASES = [
    InputWrappingTestStruct(input_infos=[ModelInputInfo([1])],
                            model_args=(TENSOR_1, ),
                            model_kwargs={},
                            ref_wrapping_sequence=[TENSOR_1],
                            ),
    InputWrappingTestStruct(input_infos=[ModelInputInfo([1], keyword="arg2")],
                            model_args=(),
                            model_kwargs={"arg2": TENSOR_2},
                            ref_wrapping_sequence=[TENSOR_2],
                            ),
    InputWrappingTestStruct(input_infos=[ModelInputInfo([1]),
                                         ModelInputInfo([1]),
                                         ModelInputInfo([1], keyword="arg3"),
                                         ModelInputInfo([1], keyword="arg5")],
                            model_args=(TENSOR_1, TENSOR_2),
                            model_kwargs={"arg3": TENSOR_3, "arg5": TENSOR_4},
                            ref_wrapping_sequence=[TENSOR_1,
                                                   TENSOR_2,
                                                   TENSOR_3,
                                                   TENSOR_4],
                            ),

    # More args supplied than what is specified by input_infos - ignore the unspecified args
    InputWrappingTestStruct(input_infos=[ModelInputInfo([1]),
                                         ModelInputInfo([1], keyword="arg3"),
                                         ModelInputInfo([1], keyword="arg5")],
                            model_args=(TENSOR_1, TENSOR_2),
                            model_kwargs={"arg3": TENSOR_3, "arg5": TENSOR_4},
                            ref_wrapping_sequence=[TENSOR_1,
                                                   TENSOR_3,
                                                   TENSOR_4],
                            ),

    # More args and kwargs supplied than what is specified by input_infos - ignore the unspecified args and kwargs
    InputWrappingTestStruct(input_infos=[ModelInputInfo([1]),
                                         ModelInputInfo([1], keyword="arg4")],
                            model_args=(TENSOR_1, TENSOR_2),
                            model_kwargs={"arg4": TENSOR_3, "arg5": TENSOR_4},
                            ref_wrapping_sequence=[TENSOR_1,
                                                   TENSOR_3],
                            ),

    # arg specified, but kwarg supplied
    InputWrappingTestStruct(input_infos=[ModelInputInfo([1])],
                            model_args=(),
                            model_kwargs={"arg3": TENSOR_1},
                            ref_wrapping_sequence=[TENSOR_FROM_INPUT_INFO_1],
                            ),

    InputWrappingTestStruct(input_infos=[ModelInputInfo([1], keyword="arg5")],
                            model_args=(),
                            model_kwargs={"arg1": TENSOR_1},
                            ref_wrapping_sequence=[TENSOR_DEFAULT],
                            ),

    # kwarg specified, but missing in supplied kwargs
    InputWrappingTestStruct(input_infos=[ModelInputInfo([1]),
                                         ModelInputInfo([2], keyword="arg3")],
                            model_args=(TENSOR_1, TENSOR_2),
                            model_kwargs={"arg4": TENSOR_3, "arg5": TENSOR_4},
                            ref_wrapping_sequence=[TENSOR_1, TENSOR_FROM_INPUT_INFO_2],
                            ),


    # More args specified than supplied
    InputWrappingTestStruct(input_infos=[ModelInputInfo([1]),
                                         ModelInputInfo([2]),
                                         ModelInputInfo([3], keyword="arg3")],
                            model_args=(TENSOR_1, ),
                            model_kwargs={"arg3": TENSOR_2},
                            ref_wrapping_sequence=[TENSOR_1, TENSOR_FROM_INPUT_INFO_2, TENSOR_2],
                            ),

    # More kwargs specified than supplied
    InputWrappingTestStruct(input_infos=[ModelInputInfo([1]),
                                         ModelInputInfo([2], keyword="arg2"),
                                         ModelInputInfo([3], keyword="arg3")],
                            model_args=(TENSOR_1,),
                            model_kwargs={"arg2": TENSOR_2},
                            ref_wrapping_sequence=[TENSOR_1, TENSOR_2, TENSOR_FROM_INPUT_INFO_3],
                            ),

]


@pytest.fixture(params=INPUT_WRAPPING_TEST_CASES, name='inputs_test_struct')
def inputs_test_struct_(request):
    return request.param


def test_input_wrapper_wrap_inputs(mocker, inputs_test_struct: InputWrappingTestStruct):
    input_infos = inputs_test_struct.input_infos
    model_args = inputs_test_struct.model_args
    model_kwargs = inputs_test_struct.model_kwargs
    ref_wrapping_sequence = inputs_test_struct.ref_wrapping_sequence
    stub_cpu_model = MockModel()

    mocker.patch('nncf.torch.dynamic_graph.io_handling.nncf_model_input')
    from nncf.torch.dynamic_graph.io_handling import nncf_model_input

    mgr = InputInfoWrapManager(input_infos, inspect.signature(forward), stub_cpu_model)
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
    config = get_basic_quantization_config(input_info=[
        {
            "sample_size": [1, 1]
        },
        {
            "sample_size": [1, 1]
        },

    ])
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
