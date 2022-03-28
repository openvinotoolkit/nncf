from nncf.torch.composite_compression import CompositeCompressionAlgorithmController
from nncf.config import NNCFConfig
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.nncf_network import EXTERNAL_QUANTIZERS_STORAGE_NAME
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.sparsity.rb.layers import RBSparsifyingWeight
from nncf.torch.utils import get_all_modules_by_type, get_all_modules
from tests.torch.helpers import BasicConvTestModel, create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args


def get_basic_sparsity_plus_quantization_config(input_sample_size=None):
    if input_sample_size is None:
        input_sample_size = [1, 1, 4, 4]
    config = NNCFConfig()
    config.update({
        "input_info":
            {
                "sample_size": input_sample_size,
            },
        "compression": [
            {
                "algorithm": "rb_sparsity",
            },
            {
                "algorithm": "quantization"
            }
        ]
    })
    return config


def test_can_quantize_inputs_for_sparsity_plus_quantization():
    model = BasicConvTestModel()
    config = get_basic_sparsity_plus_quantization_config()
    register_bn_adaptation_init_args(config)
    sparse_quantized_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert isinstance(compression_ctrl, CompositeCompressionAlgorithmController)

    sparse_quantized_model_conv = get_all_modules_by_type(sparse_quantized_model, 'NNCFConv2d')

    nncf_module = next(iter(sparse_quantized_model_conv.values()))
    assert len(nncf_module.pre_ops) == 2  # 1x weight sparsifier + 1x weight quantizer
    assert isinstance(nncf_module.pre_ops['0'], UpdateWeight)
    assert isinstance(nncf_module.pre_ops['0'].op, RBSparsifyingWeight)

    assert isinstance(nncf_module.pre_ops['1'], UpdateWeight)
    assert isinstance(nncf_module.pre_ops['1'].op, SymmetricQuantizer)

    input_quantizer = get_all_modules(
        sparse_quantized_model)[f'NNCFNetwork/ModuleDict[{EXTERNAL_QUANTIZERS_STORAGE_NAME}]']

    assert len(input_quantizer) == 1
    assert isinstance(list(input_quantizer.values())[0], SymmetricQuantizer)
