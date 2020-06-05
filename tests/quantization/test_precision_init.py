"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import itertools
import json
from collections import namedtuple, OrderedDict
from pathlib import Path
from typing import Callable

import math
import os
import pytest
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial
from torch.utils import model_zoo
from torchvision.models import MobileNetV2, mobilenet_v2
from torchvision.transforms import transforms

from examples.classification.main import create_cifar
from examples.common.model_loader import load_model
from examples.common.models import squeezenet1_1_custom
from nncf import register_default_init_args, NNCFConfig
from nncf.checkpoint_loading import load_state
from nncf.debug import set_debug_log_dir
from nncf.dynamic_graph.context import Scope, ScopeElement
from nncf.hw_config import HWConfigType
from nncf.quantization.hessian_trace import HessianTraceEstimator
from nncf.quantization.hw_precision_constraints import HWPrecisionConstraints
from nncf.quantization.init_precision import HAWQPrecisionInitializer, TracesPerLayer, Perturbations, \
    PerturbationObserver, HAWQDebugger
from nncf.quantization.layers import QUANTIZATION_MODULES, QuantizerConfig
from nncf.quantization.quantizer_id import WeightQuantizerId
from nncf.utils import get_all_modules_by_type, safe_thread_call
from tests.conftest import TEST_ROOT, EXAMPLES_DIR
from tests.quantization.test_quantization_helpers import compare_multi_gpu_dump, \
    get_quantization_config_without_range_init, distributed_init_test_default, post_compression_test_distr_init, \
    get_squeezenet_quantization_config
from tests.test_compressed_graph import check_graph
from tests.test_helpers import create_compressed_model_and_algo_for_test, MockModel, create_conv, \
    create_mock_dataloader


# pylint:disable=unused-import
from tests.modules.test_rnn import _seed


def create_test_dataloaders(model_size, dataset_dir, batch_size):
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(model_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    dummy_config = type('dummy', (object,), {'dataset_dir': dataset_dir})()
    train_dataset = create_cifar(dummy_config, dataset_config='cifar10', is_train=True, transform=train_transforms)
    pin_memory = True
    workers = 1

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                               pin_memory=pin_memory)
    return train_loader, train_dataset


def get_bitwidth_per_scope(model, all_quantizations=None):
    if not all_quantizations:
        all_quantizations = HAWQDebugger.get_all_quantizers_per_full_scope(model)
    full_bitwidth_per_scope = []
    for scope, quantizer in all_quantizations.items():
        full_bitwidth_per_scope.append([quantizer.num_bits, str(scope)])
    return full_bitwidth_per_scope


def compare_with_ref_if_exists(actual_state, path_to_ref):
    if os.path.exists(path_to_ref):
        with open(path_to_ref, 'r') as f:
            assert json.load(f) == actual_state
    else:
        with open(path_to_ref, 'w') as f:
            json.dump(actual_state, f)


def create_hawq_test_config(batch_size=10, num_data_points=100):
    config = get_quantization_config_without_range_init()
    config['input_info'] = {
        "sample_size": [1, 3, 10, 10],
    }
    config['batch_size'] = batch_size
    config['compression'].update({
        'initializer': {
            'precision': {
                "type": "hawq",
                "bits": [
                    4,
                    8,
                    6
                ],
                "num_data_points": num_data_points,
                "iter_number": 1,
                "tolerance": 1e-2
            },
            'range': {
                'num_init_steps': 1
            }
        }})
    return config


def create_hawq_hw_test_config(batch_size):
    config = create_hawq_test_config(batch_size)
    config["hw_config_type"] = HWConfigType.VPU.value
    return config


def get_avg_traces(model):
    """ Assigns bigger average traces for DepthWise Conv than for ordinary Conv and Linear"""
    all_convs = get_all_modules_by_type(model, 'Conv2d')
    dw_conv_indexes = [i for i, conv in enumerate(all_convs.values()) if conv.groups == conv.in_channels]
    dw_conv_indexes.append(len(all_convs))
    num_traces = len(all_convs) + 1  # +1 Linear

    mock_avg_traces = []
    scale = 1e-1
    device = next(model.parameters()).device
    for i in range(num_traces):
        relative_sensativity = 2 * num_traces + i if i in dw_conv_indexes else num_traces - i
        mock_avg_traces.append(torch.Tensor([scale * relative_sensativity]).to(device))
    return mock_avg_traces


def get_avg_traces_for_vpu(model):
    """
    Filters average traces for Convolutions only, as they have choice of precision on VPU and
    won't be skipped on Hessian calculation
    """
    avg_traces = get_avg_traces(model)
    all_convs = get_all_modules_by_type(model, 'Conv2d')
    return [avg_traces[i] for i, conv in enumerate(all_convs.values()) if conv.groups != conv.in_channels]


@pytest.mark.parametrize(('config_creator', 'filename_suffix', 'avg_traces_creator'),
                         ([create_hawq_test_config, 'pattern_based', get_avg_traces],
                          [create_hawq_hw_test_config, 'hw_config_vpu', get_avg_traces_for_vpu]))
def test_hawq_precision_init(_seed, dataset_dir, tmp_path, mocker, config_creator: Callable, filename_suffix: str,
                             avg_traces_creator: Callable):
    batch_size = 10
    config = config_creator(batch_size)
    model = MobileNetV2(num_classes=10)
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda()
    if not dataset_dir:
        dataset_dir = str(tmp_path)
    train_loader, _ = create_test_dataloaders(config.get("model_size"), dataset_dir, batch_size)
    config = register_default_init_args(config, criterion, train_loader)

    mocked_trace = mocker.patch('nncf.quantization.hessian_trace.HessianTraceEstimator.get_average_traces')

    mocked_trace.return_value = avg_traces_creator(model)
    from torchvision.models.mobilenet import model_urls
    load_state(model, model_zoo.load_url(model_urls['mobilenet_v2']))
    model, algo_ctrl = create_compressed_model_and_algo_for_test(model, config)
    model = model.cuda()

    all_quantizers_per_full_scope = HAWQDebugger.get_all_quantizers_per_full_scope(model)
    graph = HAWQDebugger.get_bitwidth_graph(algo_ctrl, model, all_quantizers_per_full_scope)
    path_to_dot = 'mobilenet_v2_mixed_bitwidth_graph_{}.dot'.format(filename_suffix)
    check_graph(graph, path_to_dot, os.path.join('quantized', 'hawq'), sort_dot_graph=False)


PrecisionConstraintsTestParams = namedtuple('PrecisionConstraintsTestParams',
                                            ('bits_configurations', 'constraints', 'survived_configurations_id',
                                             'order'))


def get_mock_precision_constraints(constraints, ordered_weight_keys):
    hw_precision_constraints = HWPrecisionConstraints(True)
    for key, bits in zip(ordered_weight_keys, constraints):
        bit_constraints = [QuantizerConfig(bits=bitwidth) for bitwidth in bits]
        hw_precision_constraints.add(key, bit_constraints)
    return hw_precision_constraints


def get_mock_quantizer_id(key: str) -> WeightQuantizerId:
    scope_element = ScopeElement(key)
    scope = Scope([scope_element])
    return WeightQuantizerId(scope)


@pytest.mark.parametrize("precision_constraints",
                         (
                             PrecisionConstraintsTestParams(
                                 bits_configurations=[[4, 6, 8], [8, 8, 8], [8, 6, 8], [6, 8, 8]],
                                 constraints=[[4, 8], [6], [8]],
                                 survived_configurations_id=[0, 2],
                                 order=[0, 1, 2]),
                             PrecisionConstraintsTestParams(
                                 bits_configurations=[[4, 6, 8], [8, 8, 6], [8, 8, 8]],
                                 constraints=[[6, 8], [8], [6]],
                                 survived_configurations_id=[],
                                 order=[0, 2, 1]),
                             PrecisionConstraintsTestParams(
                                 bits_configurations=[[4, 6, 8], [8, 8, 6], [8, 8, 8]],
                                 constraints=[],
                                 survived_configurations_id=[0, 1, 2],
                                 order=[0, 2, 1])
                         ))
def test_get_configs_constrained_by_precision(precision_constraints):
    bits_configurations, constraints, survived_configurations_id, order = precision_constraints
    ref_configurations = [bits_configurations[config_id] for config_id in survived_configurations_id]
    ordered_weight_keys = [get_mock_quantizer_id(str(i)) for i in range(len(constraints))]
    hw_precision_constraints = get_mock_precision_constraints(constraints, ordered_weight_keys)

    actual_configs = HAWQPrecisionInitializer._filter_configs_by_precision_constraints(
        bits_configurations, hw_precision_constraints, ordered_weight_keys, order)

    assert ref_configurations == actual_configs


HAWQTestParams = namedtuple('HAWQTestParams', ('iter_number', 'batch_size', 'num_data_points', 'ref_trace'))


@pytest.mark.parametrize("params",
                         (HAWQTestParams(200, 13, 100, 0.07957423478364944),
                          HAWQTestParams(2, 13, 100, 0.062167033553123474),
                          HAWQTestParams(2, 10, 10, 0.11200366914272308),
                          HAWQTestParams(2, 10, 5, 0.11200366914272308)),
                         ids=('until_threshold', 'until_num_iter', 'batch_eq_num_data', 'batch_larger_num_data'))
def test_hawq_on_single_conv_without_quantizers(_seed, dataset_dir, tmp_path, params: HAWQTestParams):
    config = get_squeezenet_quantization_config(batch_size=params.batch_size)
    iter_number = params.iter_number
    tolerance = 4e-4

    model = squeezenet1_1_custom(num_classes=10, pretrained=False, dropout=0)
    from examples.common.models.classification.squeezenet import model_urls
    load_state(model, model_zoo.load_url(model_urls['squeezenet1_1']))
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    if not dataset_dir:
        dataset_dir = str(tmp_path)
    data_loader, _ = create_test_dataloaders(config.get("model_size"), dataset_dir, params.batch_size)
    device = next(model.parameters()).device

    for _, param in model.named_parameters():
        param.requires_grad = False
    first_conv = next(iter(get_all_modules_by_type(model, 'Conv2d').values()))
    first_conv.weight.requires_grad = True

    trace_estimator = HessianTraceEstimator(model, criterion, device, data_loader, params.num_data_points)
    actual_state = trace_estimator.get_average_traces(max_iter=iter_number, tolerance=tolerance)
    assert math.isclose(actual_state.item(), params.ref_trace, rel_tol=1e-09)


def get_size_of_search_space(m, L):
    def nCr(n, r):
        f = math.factorial
        return f(n) // f(r) // f(n - r)

    ref_num = 0
    for j in range(1, m + 1):
        ref_num += nCr(m, j) * nCr(L - 1, j - 1)
    return ref_num


def test_constrained_bit_configs():
    bits = [4, 2, 8]
    L = 4
    m = len(bits)
    all_configs = list(itertools.product(bits, repeat=L))

    ref_configs = []
    for bit_config in all_configs:
        is_ok = True
        for i in range(L - 1):
            if bit_config[i + 1] < bit_config[i]:
                is_ok = False
                break
        if is_ok:
            ref_configs.append(list(bit_config))
    actual_config = HAWQPrecisionInitializer.get_configs_constrained_by_order(bits, L)
    ref_num = get_size_of_search_space(m, L)
    assert len(ref_configs) == ref_num
    assert len(actual_config) == ref_num
    assert sorted(actual_config) == sorted(ref_configs)


def get_requires_grad_per_param(model):
    not_sorted = OrderedDict({param_name: param.requires_grad for param_name, param in model.named_parameters()})
    return OrderedDict(sorted(not_sorted.items()))


def get_scopes_of_skipped_weight_quantizers():
    return ['MobileNetV2/Sequential[features]/ConvBNReLU[18]/NNCFConv2d[0]',
            'MobileNetV2/Sequential[features]/InvertedResidual[17]/Sequential[conv]/NNCFConv2d[2]',
            'MobileNetV2/Sequential[features]/InvertedResidual[16]/Sequential[conv]/NNCFConv2d[2]']


def test_disable_quantizer_gradients():
    _, disabled_parameters, model, _ = disable_quantizer_gradients()
    assert len(disabled_parameters) == 357
    actual_requires_grad_per_param = get_requires_grad_per_param(model)
    path_to_ref = str(TEST_ROOT / 'data/hawq_reference/mobilenet_v2_requires_grad_per_param.json')
    compare_with_ref_if_exists(actual_requires_grad_per_param, path_to_ref)


def test_enable_quantizer_gradients():
    all_quantizations, disabled_parameters, model, original_requires_grad_per_param = disable_quantizer_gradients()
    HAWQPrecisionInitializer.enable_quantizer_gradients(model, all_quantizations, disabled_parameters)
    actual_requires_grad_per_param = get_requires_grad_per_param(model)
    assert original_requires_grad_per_param == actual_requires_grad_per_param


def disable_quantizer_gradients():
    config = get_quantization_config_without_range_init()
    config['input_info'] = {
        "sample_size": [1, 3, 10, 10],
    }
    model = MobileNetV2(num_classes=10)
    model.eval()
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
    all_quantizations = get_all_modules_by_type(model, quantization_types)
    original_requires_grad_per_param = get_requires_grad_per_param(model)
    disabled_parameters = HAWQPrecisionInitializer.disable_quantizer_gradients(
        all_quantizations,
        compression_ctrl.quantized_weight_modules_registry,
        model,
        get_scopes_of_skipped_weight_quantizers())
    return all_quantizations, disabled_parameters, model, original_requires_grad_per_param


def get_path_to_bitwidth_dump(tmp_path, rank):
    out_file_path = tmp_path / 'bitwidth_per_scope_gpu{}.pt'.format(rank)
    return out_file_path


def hawq_dumping_worker(gpu, ngpus_per_node, config, tmp_path):
    data_loader = distributed_init_test_default(gpu, ngpus_per_node, config)
    model = safe_thread_call(partial(mobilenet_v2, pretrained=True))
    model.eval()
    criterion = torch.nn.MSELoss().cuda(config.gpu)
    config = register_default_init_args(config, criterion, data_loader)
    quant_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    quant_model = post_compression_test_distr_init(compression_ctrl, config, ngpus_per_node, quant_model)

    # just to reproduce the same scale values without Dropout
    quant_model.eval()

    act_bitwidth_per_scope = get_bitwidth_per_scope(quant_model.module)
    out_file_path = get_path_to_bitwidth_dump(tmp_path, config.rank)
    torch.save(act_bitwidth_per_scope, str(out_file_path))


def test_hawq_broadcast_avg_traces_in_distributed_mode(tmp_path):
    num_data_points = 100
    batch_size = 10
    config = create_hawq_test_config(batch_size, num_data_points)

    ngpus_per_node = torch.cuda.device_count()
    config.world_size = ngpus_per_node
    torch.multiprocessing.spawn(hawq_dumping_worker,
                                nprocs=ngpus_per_node,
                                args=(ngpus_per_node, config, tmp_path),
                                join=True)

    assert not compare_multi_gpu_dump(config, tmp_path, get_path_to_bitwidth_dump)


ManualConfigTestParams = namedtuple('ManualConfigTestParams', ('config_name', 'bit_stats'))
MANUAL_CONFIG_TEST_PARAMS = [
    ManualConfigTestParams(config_name="mobilenet_v2_cifar100_mixed_int_manual.json",
                           bit_stats=[['8', '23.077', '33.333', '56.410'],
                                      ['4', '22.222', '21.368', '43.590']]),
    ManualConfigTestParams(config_name="mobilenet_v2_imagenet_mixed_int_manual.json",
                           bit_stats=[['8', '23.077', '23.932', '47.009'],
                                      ['4', '22.222', '30.769', '52.991']]),
    ManualConfigTestParams(config_name="resnet50_imagenet_mixed_int_manual.json",
                           bit_stats=[['8', '21.600', '23.200', '44.800'],
                                      ['4', '21.600', '33.600', '55.200']]),
    ManualConfigTestParams(config_name="squeezenet1_1_imagenet_mixed_int_manual.json",
                           bit_stats=[['8', '24.528', '30.189', '54.717'],
                                      ['4', '24.528', '20.755', '45.283']])
]


@pytest.mark.parametrize('hw_config', [None, HWConfigType.VPU],
                         ids=['no_constraints', 'vpu_constraints'])
@pytest.mark.parametrize('manual_config_params', MANUAL_CONFIG_TEST_PARAMS,
                         ids=[pair[0] for pair in MANUAL_CONFIG_TEST_PARAMS])
def test_hawq_manual_configs(manual_config_params, hw_config):
    config_name, bit_stats = manual_config_params
    config = NNCFConfig.from_json(str(EXAMPLES_DIR.joinpath('classification', 'configs', 'quantization') / config_name))
    config = register_default_init_args(config, criterion=None, train_loader=create_mock_dataloader(config))
    if hw_config:
        config['hw_config'] = hw_config.value
    model = load_model(config['model'], pretrained=False)

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    table = compression_ctrl.get_bit_stats()
    # pylint: disable=protected-access
    assert table._rows == bit_stats


@pytest.mark.parametrize('method_name', ['_calc_traces', '_filter_configs_by_precision_constraints'])
def test_hawq_raises_error_if_method_returns_none(mocker, method_name):
    config = create_hawq_test_config()
    model = MockModel()
    config = register_default_init_args(config, mocker.stub(), mocker.stub())
    mocker.patch('nncf.quantization.algo.QuantizationController._do_range_init')
    mocker.patch('nncf.quantization.init_precision.HAWQPrecisionInitializer._calc_traces')

    mocked_trace = mocker.patch('nncf.quantization.init_precision.HAWQPrecisionInitializer.' + method_name)
    mocked_trace.return_value = None

    with pytest.raises(RuntimeError):
        create_compressed_model_and_algo_for_test(model, config)


def test_check_hawq_dump(mocker, tmp_path):
    tensor1 = torch.Tensor([1])
    tensor2 = torch.Tensor([2])
    bitwidth1 = 2
    bitwidth2 = 4
    id = 0
    bits_configurations = [[bitwidth1], [bitwidth2]]
    configuration_metric = [tensor1, tensor2]
    perturbations = Perturbations()
    perturbations.add(id, bitwidth1, tensor1)
    perturbations.add(id, bitwidth2, tensor2)
    observer = PerturbationObserver(mocker.stub())
    observer.perturbation = tensor1
    observer.numels = id
    observer.input_norm = id
    weight_observers = [observer]
    traces_per_layer = TracesPerLayer(torch.cat((tensor1, tensor2)))

    set_debug_log_dir(str(tmp_path))
    hawq_debugger = HAWQDebugger(bits_configurations, perturbations, weight_observers, traces_per_layer,
                                 [bitwidth1, bitwidth2])

    hawq_debugger.dump_metric(configuration_metric)
    hawq_debugger.dump_avg_traces()
    hawq_debugger.dump_density_of_quantization_noise()
    hawq_debugger.dump_perturbations_ratio()
    test_dir = tmp_path / Path('hawq_dumps')
    num_dump_files = len([name for name in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, name))])
    assert num_dump_files == 5


#       fq_2
#        \
# fq_2 - conv_1 - fq_6
#                   \
#        fq_4       add
#         \         /
# fq_4 - conv_2 - fq_6
#
def test_quantization_configs__with_precisions_list():
    class ModelForTest(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = create_conv(1, 2, 2, -1, -2)
            self.conv2 = create_conv(1, 2, 2, -1, -2)

        def forward(self, x):
            return self.conv1(x) + self.conv2(x)

    model = ModelForTest()

    config = get_quantization_config_without_range_init()
    config['compression']['initializer'].update({
        "precision": {
            "bitwidth_per_scope":
                [[2, 'ModelForTest/NNCFConv2d[conv1]'],
                 [4, 'ModelForTest/NNCFConv2d[conv2]']]
        }})
    config['compression']["activations"] = {"bits": 6}

    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    ref_bits = [('ModelForTest/NNCFConv2d[conv1]module_weight', 2),
                ('ModelForTest/NNCFConv2d[conv2]module_weight', 4),
                ('ModelForTest/NNCFConv2d[conv2]/conv2d_0', 6),
                ('ModelForTest/NNCFConv2d[conv1]/conv2d_0', 6),
                ('ModelForTest/NNCFConv2d[conv1]module_input', 2),
                ('ModelForTest/NNCFConv2d[conv2]module_input', 4)]

    for key, quantizer in compression_ctrl.all_quantizations.items():
        expected_bit = [ref_bit for (name, ref_bit) in ref_bits if name == str(key)][0]
        assert quantizer.num_bits == expected_bit, 'Unexpected number of bits for {}'.format(key)

    ref_rows = [['2', '16.667', '16.667', '33.333'],
                ['4', '16.667', '16.667', '33.333'],
                ['6', '0', '33.333', '33.333']]
    table = compression_ctrl.get_bit_stats()
    # pylint: disable=protected-access
    assert table._rows == ref_rows
