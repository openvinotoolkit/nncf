import pytest
from pytest import approx
from collections import namedtuple

from nncf.quantization.metrics import NetworkQuantizationShareMetric as NQSM,\
    MemoryCostMetric, ShareEdgesQuantizedDataPath
from nncf import create_compressed_model, NNCFConfig
from tests import test_models

def get_basic_quantization_config():
    config = NNCFConfig()
    config.update({
        "model": "AlexNet",
        "input_info":
            {
                "sample_size": [1, 3, 32, 32],
            },
        "compression":
            {
                "algorithm": "quantization",
                "quantize_inputs": True,
                "initializer": {
                    "range": {
                        "num_init_samples": 0
                    }
                }
            }
    })

    return config

TestStruct = namedtuple('TestStruct',
                        ('initializers',
                         'activations',
                         'weights',
                         'ignored_scopes',
                         'table',
                         'target_device',
                         'quantizer_setup_type'))

NETWORK_QUANTIZATION_SHARE_METRIC_TEST_CASES = [
    TestStruct(
        initializers={},
        activations={},
        weights={},
        ignored_scopes={},
        target_device='TRIAL',
        quantizer_setup_type='pattern_based',
        table={NQSM.ACTIVATIONS_RATIO_STR: {NQSM.UNSIGNED_STR: 100.0, NQSM.PER_CHANNEL_STR: 0.0,
                                            NQSM.SYMMETRIC_STR: 100.0, NQSM.ASYMMETRIC_STR: 0.0,
                                            NQSM.PER_TENSOR_STR: 100.0, NQSM.SIGNED_STR: 0.0, 8: 100.0},\
               NQSM.TOTAL_RATIO_STR: {NQSM.UNSIGNED_STR: 0.0, NQSM.PER_CHANNEL_STR: 0.0, NQSM.SYMMETRIC_STR: 0.0,\
               NQSM.ASYMMETRIC_STR: 0.0, NQSM.PER_TENSOR_STR: 0.0, NQSM.SIGNED_STR: 0.0, 8: 100.0},\
               NQSM.WEIGHTS_RATIO_STR: {NQSM.UNSIGNED_STR: 100.0, NQSM.PER_CHANNEL_STR: 0.0, NQSM.SYMMETRIC_STR: 100.0,\
               NQSM.ASYMMETRIC_STR: 0.0, NQSM.PER_TENSOR_STR: 100.0, NQSM.SIGNED_STR: 0.0, 8: 100.0},\
              'Quantizer parameter': {NQSM.UNSIGNED_STR: 0, NQSM.PER_CHANNEL_STR: 0, NQSM.SYMMETRIC_STR: 0,\
               NQSM.ASYMMETRIC_STR: 0, NQSM.PER_TENSOR_STR: 0, NQSM.SIGNED_STR: 0, 8: 0}}),
    TestStruct(
        initializers={},
        activations={},
        weights={},
        ignored_scopes={},
        target_device='TRIAL',
        quantizer_setup_type='propagation_based',
        table={NQSM.ACTIVATIONS_RATIO_STR: {NQSM.SIGNED_STR: 0.0, NQSM.PER_CHANNEL_STR: 0.0, NQSM.UNSIGNED_STR: 100.0,\
               NQSM.PER_TENSOR_STR: 100.0, NQSM.SYMMETRIC_STR: 100.0, NQSM.ASYMMETRIC_STR: 0.0, 8: 100.0},\
               NQSM.TOTAL_RATIO_STR: {NQSM.SIGNED_STR: 0.0, NQSM.PER_CHANNEL_STR: 0.0, NQSM.UNSIGNED_STR: 0.0,\
               NQSM.PER_TENSOR_STR: 0.0, NQSM.SYMMETRIC_STR: 0.0, NQSM.ASYMMETRIC_STR: 0.0, 8: 100.0},\
               NQSM.WEIGHTS_RATIO_STR: {NQSM.SIGNED_STR: 0.0, NQSM.PER_CHANNEL_STR: 0.0, NQSM.UNSIGNED_STR: 100.0,\
               NQSM.PER_TENSOR_STR: 100.0, NQSM.SYMMETRIC_STR: 100.0, NQSM.ASYMMETRIC_STR: 0.0, 8: 100.0},\
               'Quantizer parameter': {NQSM.SIGNED_STR: 0, NQSM.PER_CHANNEL_STR: 0, NQSM.UNSIGNED_STR: 0,\
               NQSM.PER_TENSOR_STR: 0, NQSM.SYMMETRIC_STR: 0, NQSM.ASYMMETRIC_STR: 0, 8: 0}}
    ),
    TestStruct(
        initializers={},
        activations={},
        weights={},
        ignored_scopes=[],
        target_device='TRIAL',
        quantizer_setup_type='pattern_based',
        table={NQSM.ACTIVATIONS_RATIO_STR: {NQSM.SIGNED_STR: 0.0, NQSM.PER_CHANNEL_STR: 0.0, NQSM.UNSIGNED_STR: 100.0,\
               NQSM.PER_TENSOR_STR: 100.0, NQSM.SYMMETRIC_STR: 100.0, NQSM.ASYMMETRIC_STR: 0.0, 8: 100.0},\
               NQSM.TOTAL_RATIO_STR: {NQSM.SIGNED_STR: 0.0, NQSM.PER_CHANNEL_STR: 0.0, NQSM.UNSIGNED_STR: 0.0,\
               NQSM.PER_TENSOR_STR: 0.0, NQSM.SYMMETRIC_STR: 0.0, NQSM.ASYMMETRIC_STR: 0.0, 8: 100.0},\
               NQSM.WEIGHTS_RATIO_STR: {NQSM.SIGNED_STR: 0.0, NQSM.PER_CHANNEL_STR: 0.0,\
               NQSM.UNSIGNED_STR: 100.0, NQSM.PER_TENSOR_STR: 100.0, NQSM.SYMMETRIC_STR: 100.0,\
               NQSM.ASYMMETRIC_STR: 0.0, 8: 100.0},\
               'Quantizer parameter': {NQSM.SIGNED_STR: 0, NQSM.PER_CHANNEL_STR: 0, NQSM.UNSIGNED_STR: 0,\
               NQSM.PER_TENSOR_STR: 0, NQSM.SYMMETRIC_STR: 0, NQSM.ASYMMETRIC_STR: 0, 8: 0}}
    ),
    TestStruct(
        initializers={},
        activations={},
        weights={},
        ignored_scopes=[],
        target_device='CPU',
        quantizer_setup_type='propagation_based',
        table={NQSM.ACTIVATIONS_RATIO_STR: {NQSM.SIGNED_STR: 0.0, NQSM.PER_CHANNEL_STR: 0.0, NQSM.UNSIGNED_STR: 100.0,\
               NQSM.PER_TENSOR_STR: 100.0, NQSM.SYMMETRIC_STR: 100.0, NQSM.ASYMMETRIC_STR: 0.0, 8: 100.0},\
               NQSM.TOTAL_RATIO_STR: {NQSM.SIGNED_STR: 0.0, NQSM.PER_CHANNEL_STR: 0.0, NQSM.UNSIGNED_STR: 0.0,\
               NQSM.PER_TENSOR_STR: 0.0, NQSM.SYMMETRIC_STR: 0.0, NQSM.ASYMMETRIC_STR: 0.0, 8: 100.0},\
               NQSM.WEIGHTS_RATIO_STR: {NQSM.SIGNED_STR: 100.0, NQSM.PER_CHANNEL_STR: 100.0, NQSM.UNSIGNED_STR: 0.0,\
               NQSM.PER_TENSOR_STR: 0.0, NQSM.SYMMETRIC_STR: 100.0, NQSM.ASYMMETRIC_STR: 0.0, 8: 100.0},\
               'Quantizer parameter': {NQSM.SIGNED_STR: 0, NQSM.PER_CHANNEL_STR: 0, NQSM.UNSIGNED_STR: 0,\
               NQSM.PER_TENSOR_STR: 0, NQSM.SYMMETRIC_STR: 0, NQSM.ASYMMETRIC_STR: 0, 8: 0}}
    ),
    TestStruct(
        initializers={"precision": {
            "bitwidth_per_scope":
            [[2, 'AlexNet/Sequential[features]/NNCFConv2d[0]'],
             [4, 'AlexNet/Sequential[features]/NNCFConv2d[6]']]
            }},
        activations={
            "mode": "asymmetric",
            "bits": 8,
            "signed": True,
            },
        weights={
            "mode": "symmetric",
            "per_channel": True,
            "bits": 8
            },
        ignored_scopes=['AlexNet/Sequential[classifier]'],
        quantizer_setup_type='pattern_based',
        target_device='TRIAL',
        table={NQSM.ACTIVATIONS_RATIO_STR: {NQSM.SIGNED_STR: 100.0, NQSM.PER_CHANNEL_STR: 0.0, NQSM.UNSIGNED_STR: 0.0,\
               NQSM.PER_TENSOR_STR: 100.0, NQSM.SYMMETRIC_STR: 0.0, NQSM.ASYMMETRIC_STR: 100.0, 8: 100.0, 2: 0,\
               4: 0.0}, NQSM.TOTAL_RATIO_STR: {NQSM.SIGNED_STR: 0.0, NQSM.PER_CHANNEL_STR: 0.0, NQSM.UNSIGNED_STR: 0.0,\
               NQSM.PER_TENSOR_STR: 0.0, NQSM.SYMMETRIC_STR: 0.0,\
               NQSM.ASYMMETRIC_STR: 0.0, 8: 81.81, 2: 9.09, 4: 9.09},\
               NQSM.WEIGHTS_RATIO_STR: {NQSM.SIGNED_STR: 0.0, NQSM.PER_CHANNEL_STR: 100.0, NQSM.UNSIGNED_STR: 100.0,\
               NQSM.PER_TENSOR_STR: 0.0, NQSM.SYMMETRIC_STR: 100.0,\
               NQSM.ASYMMETRIC_STR: 0.0, 8: 60.0, 2: 20.0, 4: 20.0},\
               'Quantizer parameter': {NQSM.SIGNED_STR: 0, NQSM.PER_CHANNEL_STR: 0, NQSM.UNSIGNED_STR: 0,\
               NQSM.PER_TENSOR_STR: 0, NQSM.SYMMETRIC_STR: 0, NQSM.ASYMMETRIC_STR: 0, 8: 0, 2: 0, 4: 0}}
    ),
    TestStruct(
        initializers={"precision": {
            "bitwidth_per_scope":
            [[2, 'AlexNet/Sequential[features]/NNCFConv2d[0]'],
             [4, 'AlexNet/Sequential[features]/NNCFConv2d[6]']]
        }},
        activations={"bits": 8},
        weights={"bits":6},
        ignored_scopes=['AlexNet/Sequential[classifier]'],
        target_device='TRIAL',
        quantizer_setup_type='pattern_based',
        table={'Quantizer parameter': {NQSM.SIGNED_STR: 0, NQSM.PER_CHANNEL_STR: 0, NQSM.UNSIGNED_STR: 0,\
               NQSM.PER_TENSOR_STR: 0, NQSM.SYMMETRIC_STR: 0, NQSM.ASYMMETRIC_STR: 0, 8: 0, 2: 0, 4: 0, 6: 0},\
               NQSM.WEIGHTS_RATIO_STR: {NQSM.SIGNED_STR: 0.0, NQSM.PER_CHANNEL_STR: 0.0, NQSM.UNSIGNED_STR: 100.0,\
               NQSM.PER_TENSOR_STR: 100.0, NQSM.SYMMETRIC_STR: 100.0, NQSM.ASYMMETRIC_STR: 0.0, 8: 0.0, 2: 20.0,\
               4: 20.0, 6: 60.0}, NQSM.ACTIVATIONS_RATIO_STR: {NQSM.SIGNED_STR: 0.0, NQSM.PER_CHANNEL_STR: 0.0,\
               NQSM.UNSIGNED_STR: 100.0, NQSM.PER_TENSOR_STR: 100.0,\
               NQSM.SYMMETRIC_STR: 100.0, NQSM.ASYMMETRIC_STR: 0.0,\
               8: 100.00, 2: 0, 4: 0.0, 6: 0.0}, NQSM.TOTAL_RATIO_STR: {NQSM.SIGNED_STR: 0.0,\
               NQSM.PER_CHANNEL_STR: 0.0, NQSM.UNSIGNED_STR: 0.0, NQSM.PER_TENSOR_STR: 0.0, NQSM.SYMMETRIC_STR: 0.0,\
               NQSM.ASYMMETRIC_STR: 0.0, 8: 54.54, 2: 9.09, 4: 9.09, 6: 27.27}}
    ),
    TestStruct(
        initializers={"precision": {
            "bitwidth_per_scope":
            [[2, 'AlexNet/Sequential[features]/NNCFConv2d[0]'],
             [4, 'AlexNet/Sequential[features]/NNCFConv2d[6]']]
        }},
        activations={},
        weights={"bits":6},
        ignored_scopes=[],
        target_device='TRIAL',
        quantizer_setup_type='pattern_based',
        table={'Quantizer parameter': {NQSM.SIGNED_STR: 0, NQSM.PER_CHANNEL_STR: 0, NQSM.UNSIGNED_STR: 0,\
               NQSM.PER_TENSOR_STR: 0, NQSM.SYMMETRIC_STR: 0, NQSM.ASYMMETRIC_STR: 0, 8: 0, 2: 0, 4: 0, 6: 0},\
               NQSM.WEIGHTS_RATIO_STR: {NQSM.SIGNED_STR: 0.0, NQSM.PER_CHANNEL_STR: 0.0, NQSM.UNSIGNED_STR: 100.0,\
               NQSM.PER_TENSOR_STR: 100.0, NQSM.SYMMETRIC_STR: 100.0, NQSM.ASYMMETRIC_STR: 0.0, 8: 0.0, 2: 12.5,\
               4: 12.5, 6: 75.0}, NQSM.ACTIVATIONS_RATIO_STR: {NQSM.SIGNED_STR: 0.0, NQSM.PER_CHANNEL_STR: 0.0,\
               NQSM.UNSIGNED_STR: 100.0, NQSM.PER_TENSOR_STR: 100.0, NQSM.SYMMETRIC_STR: 100.0,\
               NQSM.ASYMMETRIC_STR: 0.0, 8: 100, 2: 0, 4: 0.0, 6: 0.0},\
               NQSM.TOTAL_RATIO_STR: {NQSM.SIGNED_STR: 0.0, NQSM.PER_CHANNEL_STR: 0.0, NQSM.UNSIGNED_STR: 0.0,\
               NQSM.PER_TENSOR_STR: 0.0, NQSM.SYMMETRIC_STR: 0.0, NQSM.ASYMMETRIC_STR: 0.0,\
               8: 50.0, 2: 6.25, 4: 6.25, 6: 37.5}}
    )
]


@pytest.fixture(params=NETWORK_QUANTIZATION_SHARE_METRIC_TEST_CASES)
def network_quantization_share_metric_test_struct(request):
    return request.param

# pylint: disable=redefined-outer-name
def test_network_quantization_share_metric(network_quantization_share_metric_test_struct):
    config = get_basic_quantization_config()
    config['compression']['initializer'].update(network_quantization_share_metric_test_struct.initializers)
    config['compression']["activations"] = network_quantization_share_metric_test_struct.activations
    config['compression']["weights"] = network_quantization_share_metric_test_struct.weights
    config['compression']["ignored_scopes"] = network_quantization_share_metric_test_struct.ignored_scopes
    config['quantizer_setup_type'] = network_quantization_share_metric_test_struct.quantizer_setup_type
    config['target_device'] = network_quantization_share_metric_test_struct.target_device
    ctrl, _ = create_compressed_model(test_models.AlexNet(), config)
    qmetric = ctrl.non_stable_metric_collectors[0]
    qmetric.collect()
    # pylint: disable=protected-access
    qmetric_stat = qmetric._get_copy_statistics()
    for key, value in network_quantization_share_metric_test_struct.table.items():
        assert qmetric_stat[key] == approx(value, rel=1e-2)


MEMORY_COST_METRIC_TEST_CASES = [
    TestStruct(
        initializers={},
        activations={},
        weights={},
        ignored_scopes=[],
        target_device='TRIAL',
        quantizer_setup_type='pattern_based',
        table={MemoryCostMetric.EXPECTED_MEMORY_CONSUMPTION_DECREASE_STR: 4.0,
               MemoryCostMetric.SIZE_MEMORY_FP_WEIGHTS_STR: 88.74,
               MemoryCostMetric.SIZE_MEMORY_COMPRESSED_WEIGHTS_STR: 22.18,
               MemoryCostMetric.MAX_MEMORY_CONSUMPTION_ACTIVATION_TENSOR_IN_COMPRESSED_MODEL_STR: 0.0625,
               MemoryCostMetric.MAX_MEMORY_CONSUMPTION_ACTIVATION_TENSOR_IN_FP32_MODEL_STR: 0.0625}),
    TestStruct(
        initializers={"precision": {
            "bitwidth_per_scope":
            [[2, 'AlexNet/Sequential[features]/NNCFConv2d[0]'],
             [4, 'AlexNet/Sequential[features]/NNCFConv2d[6]']]
        }},
        activations={},
        weights={"bits": 8},
        ignored_scopes=[],
        target_device='TRIAL',
        quantizer_setup_type='pattern_based',
        table={MemoryCostMetric.EXPECTED_MEMORY_CONSUMPTION_DECREASE_STR: 4.05,
               MemoryCostMetric.SIZE_MEMORY_FP_WEIGHTS_STR: 88.74,
               MemoryCostMetric.SIZE_MEMORY_COMPRESSED_WEIGHTS_STR: 21.86,
               MemoryCostMetric.MAX_MEMORY_CONSUMPTION_ACTIVATION_TENSOR_IN_COMPRESSED_MODEL_STR: 0.0625,
               MemoryCostMetric.MAX_MEMORY_CONSUMPTION_ACTIVATION_TENSOR_IN_FP32_MODEL_STR: 0.0625}),
    TestStruct(
        initializers={},
        activations={},
        weights={},
        ignored_scopes=['AlexNet/Sequential[features]/NNCFConv2d[0]'],
        target_device='TRIAL',
        quantizer_setup_type='pattern_based',
        table={MemoryCostMetric.EXPECTED_MEMORY_CONSUMPTION_DECREASE_STR: 3.99,
               MemoryCostMetric.SIZE_MEMORY_FP_WEIGHTS_STR: 88.74,
               MemoryCostMetric.SIZE_MEMORY_COMPRESSED_WEIGHTS_STR: 22.19,
               MemoryCostMetric.MAX_MEMORY_CONSUMPTION_ACTIVATION_TENSOR_IN_COMPRESSED_MODEL_STR: 0.0625,
               MemoryCostMetric.MAX_MEMORY_CONSUMPTION_ACTIVATION_TENSOR_IN_FP32_MODEL_STR: 0.0625}),
]

@pytest.fixture(params=MEMORY_COST_METRIC_TEST_CASES)
def memory_cost_metric_test_struct(request):
    return request.param

# pylint: disable=redefined-outer-name
def test_memory_cost_metric(memory_cost_metric_test_struct):
    config = get_basic_quantization_config()
    config['compression']['initializer'].update(memory_cost_metric_test_struct.initializers)
    config['compression']["weights"] = memory_cost_metric_test_struct.weights
    config['compression']["ignored_scopes"] = memory_cost_metric_test_struct.ignored_scopes
    config['target_device'] = memory_cost_metric_test_struct.target_device
    ctrl, compressed_model = create_compressed_model(test_models.AlexNet(), config)
    qmetric = MemoryCostMetric(compressed_model, ctrl.weight_quantizers, ctrl.non_weight_quantizers)
    qmetric.collect()

    assert qmetric.stat == approx(memory_cost_metric_test_struct.table, rel=1e-2)

SHARE_EDGES_QUANTIZED_DATA_PATH_TEST_CASES = [
    TestStruct(
        initializers={},
        activations={},
        weights={},
        ignored_scopes=[],
        target_device='TRIAL',
        quantizer_setup_type='propagation_based',
        table={ShareEdgesQuantizedDataPath.COUNT_QUANTIZED_EDGES_STR: 100}
    ),
    TestStruct(
        initializers={},
        activations={},
        weights={},
        ignored_scopes=[],
        target_device='TRIAL',
        quantizer_setup_type='pattern_based',
        table={ShareEdgesQuantizedDataPath.COUNT_QUANTIZED_EDGES_STR: 100}
    ),
    TestStruct(
        initializers={},
        activations={},
        weights={},
        ignored_scopes=[
            "Inception3/__add___0",
            "Inception3/__add___1",
            "Inception3/__add___2",
            "Inception3/__mul___0",
            "Inception3/__mul___1",
            "Inception3/__mul___2"],
        target_device='TRIAL',
        quantizer_setup_type='pattern_based',
        table={ShareEdgesQuantizedDataPath.COUNT_QUANTIZED_EDGES_STR: 95.97}
    )
]

@pytest.fixture(params=SHARE_EDGES_QUANTIZED_DATA_PATH_TEST_CASES)
def share_edges_quantized_data_path_test_struct(request):
    return request.param

# pylint: disable=redefined-outer-name
def test_share_edges_quantized_data_path(share_edges_quantized_data_path_test_struct):
    config = get_basic_quantization_config()
    config['compression']["ignored_scopes"] = share_edges_quantized_data_path_test_struct.ignored_scopes
    config['input_info']['sample_size'] = [2, 3, 299, 299]
    config['quantizer_setup_type'] = share_edges_quantized_data_path_test_struct.quantizer_setup_type

    _, compressed_model = create_compressed_model(test_models.Inception3(aux_logits=True, transform_input=True), config)
    qmetric = ShareEdgesQuantizedDataPath(compressed_model)
    qmetric.collect()
    # pylint: disable=protected-access
    qmetric_stat = qmetric._get_copy_statistics()
    assert qmetric_stat == approx(share_edges_quantized_data_path_test_struct.table, rel=1e-2)
