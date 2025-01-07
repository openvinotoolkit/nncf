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

from nncf.config import NNCFConfig
from nncf.torch.composite_compression import CompositeCompressionAlgorithmController
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.quantization.external_quantizer import EXTERNAL_QUANTIZERS_STORAGE_NAME
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.sparsity.rb.layers import RBSparsifyingWeight
from nncf.torch.utils import get_all_modules
from nncf.torch.utils import get_all_modules_by_type
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args


def get_basic_sparsity_plus_quantization_config(input_sample_size=None):
    if input_sample_size is None:
        input_sample_size = [1, 1, 4, 4]
    config = NNCFConfig()
    config.update(
        {
            "input_info": {
                "sample_size": input_sample_size,
            },
            "compression": [
                {
                    "algorithm": "rb_sparsity",
                },
                {"algorithm": "quantization"},
            ],
        }
    )
    return config


def test_can_quantize_inputs_for_sparsity_plus_quantization():
    model = BasicConvTestModel()
    config = get_basic_sparsity_plus_quantization_config()
    register_bn_adaptation_init_args(config)
    sparse_quantized_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert isinstance(compression_ctrl, CompositeCompressionAlgorithmController)

    sparse_quantized_model_conv = get_all_modules_by_type(sparse_quantized_model, "NNCFConv2d")

    nncf_module = next(iter(sparse_quantized_model_conv.values()))
    assert len(nncf_module.pre_ops) == 2  # 1x weight sparsifier + 1x weight quantizer
    assert isinstance(nncf_module.pre_ops["0"], UpdateWeight)
    assert isinstance(nncf_module.pre_ops["0"].op, RBSparsifyingWeight)

    assert isinstance(nncf_module.pre_ops["1"], UpdateWeight)
    assert isinstance(nncf_module.pre_ops["1"].op, SymmetricQuantizer)

    input_quantizer = get_all_modules(sparse_quantized_model)[
        f"BasicConvTestModel/" f"NNCFNetworkInterface[_nncf]/ModuleDict[{EXTERNAL_QUANTIZERS_STORAGE_NAME}]"
    ]

    assert len(input_quantizer) == 1
    assert isinstance(list(input_quantizer.values())[0], SymmetricQuantizer)


def test_compression_rate_for_sparsity_plus_quantization():
    model = BasicConvTestModel()
    config = get_basic_sparsity_plus_quantization_config()
    register_bn_adaptation_init_args(config)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert compression_ctrl.compression_rate == 0.0
