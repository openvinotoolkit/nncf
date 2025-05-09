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
from copy import deepcopy

import pytest
import torch

from nncf.torch.checkpoint_loading import load_state
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.sparsity.const.algo import ConstSparsityController
from nncf.torch.sparsity.layers import BinaryMask
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import PTTensorListComparator
from tests.torch.helpers import check_correct_nncf_modules_replacement
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import get_empty_config
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.sparsity.magnitude.test_helpers import MagnitudeTestModel

pytestmark = pytest.mark.legacy

sub_tensor = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
ref_mask_1 = torch.cat((sub_tensor, sub_tensor), 0)
sub_tensor = torch.tensor([[[[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]]])
ref_mask_2 = torch.cat((sub_tensor, sub_tensor), 1)


def test_can_create_const_sparse_algo__with_default():
    model = BasicConvTestModel()
    config = get_empty_config()
    config["compression"] = {"algorithm": "const_sparsity"}
    sparse_model, compression_ctrl = create_compressed_model_and_algo_for_test(deepcopy(model), config)

    assert isinstance(compression_ctrl, ConstSparsityController)
    assert len(list(sparse_model.modules())) == 7

    _, sparse_model_conv = check_correct_nncf_modules_replacement(model, sparse_model)

    for sparse_module in sparse_model_conv.values():
        store = []
        for op in sparse_module.pre_ops.values():
            if isinstance(op, UpdateWeight) and isinstance(op.operand, BinaryMask):
                ref_mask = torch.ones_like(sparse_module.weight)
                assert torch.allclose(op.operand.binary_mask, ref_mask)
                assert op.__class__.__name__ not in store
                store.append(op.__class__.__name__)


def test_can_restore_binary_mask_on_magnitude_algo_resume():
    config = get_empty_config()
    config["compression"] = {
        "algorithm": "magnitude_sparsity",
        "params": {"weight_importance": "abs", "schedule": "multistep", "multistep_sparsity_levels": [0.3, 0.5]},
    }

    sparse_model, _ = create_compressed_model_and_algo_for_test(MagnitudeTestModel(), config)
    with torch.no_grad():
        sparse_model(torch.ones([1, 1, 10, 10]))

    config = get_empty_config()
    config["compression"] = {"algorithm": "const_sparsity"}
    const_sparse_model, _ = create_compressed_model_and_algo_for_test(MagnitudeTestModel(), config)

    load_state(const_sparse_model, sparse_model.state_dict())

    op = const_sparse_model.conv1.pre_ops["0"]
    PTTensorListComparator.check_equal(ref_mask_1, op.operand.binary_mask)

    op = const_sparse_model.conv2.pre_ops["0"]
    PTTensorListComparator.check_equal(ref_mask_2, op.operand.binary_mask)


@pytest.mark.parametrize(
    "use_data_parallel", [pytest.param(True, marks=pytest.mark.cuda), False], ids=["dataparallel", "regular"]
)
def test_can_restore_binary_mask_on_magnitude_quant_algo_resume(tmp_path, use_data_parallel):
    config = get_empty_config()
    config["compression"] = [
        {
            "algorithm": "magnitude_sparsity",
            "params": {"schedule": "multistep", "multistep_sparsity_levels": [0.3, 0.5], "weight_importance": "abs"},
        },
        {"algorithm": "quantization"},
    ]
    register_bn_adaptation_init_args(config)

    sparse_model, _ = create_compressed_model_and_algo_for_test(MagnitudeTestModel(), config)

    if use_data_parallel:
        if not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test cases for CPU only setups")
        sparse_model.cuda()
        sparse_model = torch.nn.DataParallel(sparse_model)

    with torch.no_grad():
        sparse_model(torch.ones([1, 1, 10, 10]))

    config = get_empty_config()
    config["compression"] = [{"algorithm": "const_sparsity"}, {"algorithm": "quantization"}]
    register_bn_adaptation_init_args(config)
    const_sparse_model, _ = create_compressed_model_and_algo_for_test(MagnitudeTestModel(), config)

    load_state(const_sparse_model, sparse_model.state_dict())

    op = const_sparse_model.conv1.pre_ops["0"]
    PTTensorListComparator.check_equal(ref_mask_1, op.operand.binary_mask)

    op = const_sparse_model.conv2.pre_ops["0"]
    PTTensorListComparator.check_equal(ref_mask_2, op.operand.binary_mask)
