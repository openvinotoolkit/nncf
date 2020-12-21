import copy

from nncf import register_default_init_args, NNCFConfig
from nncf.automl.environment.quantization_env import QuantizationEnv
from tests.helpers import create_mock_dataloader, BasicConvTestModel, create_compressed_model_and_algo_for_test
from tests.quantization.test_precision_init import AutoQConfigBuilder

# TODO: temporary value to be replaced with actual one
from tests.quantization.test_quantization_helpers import get_quantization_config_without_range_init

STUB = 0


def create_test_quantization_env() -> QuantizationEnv:
    model = BasicConvTestModel()
    config = get_quantization_config_without_range_init()
    config['target_device'] = 'VPU'
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    config['compression']['initializer']['precision'] = {"type": "autoq"}
    data_loader = create_mock_dataloader(config['input_info'])
    return QuantizationEnv(compression_ctrl, data_loader, lambda *x: 0, config)


def test_can_create_quant_env():
    create_test_quantization_env()


def test_step():
    qenv = create_test_quantization_env()
    action = STUB
    ref_result = STUB
    actual_result = qenv.step(action)
    assert actual_result == ref_result


def test_reward():
    qenv = create_test_quantization_env()
    acc = STUB
    model_ratio = STUB
    qenv.reward(acc, model_ratio)


def test_evaluate_strategy():
    qenv = create_test_quantization_env()
    collected_strategy = [STUB]
    obs, reward, done, info_set = qenv.evaluate_strategy(collected_strategy)
    assert obs == STUB
    assert reward == STUB
    assert done == STUB
    assert info_set == STUB
