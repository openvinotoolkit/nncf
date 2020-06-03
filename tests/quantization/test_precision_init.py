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
from examples.common.models import squeezenet1_1_custom
from nncf import register_default_init_args
from nncf.checkpoint_loading import load_state
from nncf.dynamic_graph.context import Scope, ScopeElement
from nncf.dynamic_graph.graph import NNCFGraph
from nncf.dynamic_graph.graph_builder import create_input_infos
from nncf.hw_config import HWConfigType
from nncf.layers import NNCFConv2d
from nncf.nncf_network import CompressionModuleType
from nncf.quantization.hessian_trace import HessianTraceEstimator
from nncf.quantization.hw_precision_constraints import HWPrecisionConstraints
from nncf.quantization.init_precision import HAWQPrecisionInitializer
from nncf.quantization.layers import QUANTIZATION_MODULES, QuantizerConfig
from nncf.quantization.quantizer_id import WeightQuantizerId
from nncf.utils import get_all_modules_by_type, safe_thread_call
from tests.conftest import TEST_ROOT
from tests.quantization.test_algo_quantization import get_squeezenet_quantization_config, \
    get_basic_quantization_config, RankDatasetMock, compare_multi_gpu_dump
from tests.test_compressed_graph import check_graph
from tests.test_helpers import create_compressed_model_and_algo_for_test

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
        all_quantizations = get_all_quantizers_per_full_scope(model)
    full_bitwidth_per_scope = []
    for scope, quantizer in all_quantizations.items():
        full_bitwidth_per_scope.append([quantizer.num_bits, str(scope)])
    return full_bitwidth_per_scope


def get_all_quantizers_per_full_scope(model):
    all_quantizations = OrderedDict()
    for class_type in QUANTIZATION_MODULES.registry_dict.values():
        quantization_type = class_type.__name__
        all_quantizations.update(
            get_all_modules_by_type(model.get_compression_modules_by_type(CompressionModuleType.ACTIVATION_QUANTIZER),
                                    quantization_type))
        all_quantizations.update(
            get_all_modules_by_type(model.get_compression_modules_by_type(CompressionModuleType.FUNCTION_QUANTIZER),
                                    quantization_type))
        all_quantizations.update(get_all_modules_by_type(model.get_nncf_wrapped_model(), quantization_type))
    all_quantizations = OrderedDict(sorted(all_quantizations.items(), key=lambda x: str(x[0])))
    return all_quantizations


def compare_with_ref_if_exists(actual_state, path_to_ref):
    if os.path.exists(path_to_ref):
        with open(path_to_ref, 'r') as f:
            assert json.load(f) == actual_state
    else:
        with open(path_to_ref, 'w') as f:
            json.dump(actual_state, f)


def create_hawq_test_config(batch_size, num_data_points):
    config = get_basic_quantization_config()
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


def create_hawq_hw_test_config(batch_size, num_data_points):
    config = create_hawq_test_config(batch_size, num_data_points)
    config["hw_config_type"] = HWConfigType.VPU.value
    return config


@pytest.mark.parametrize(('config_creator', 'filename_suffix'),
                         ([create_hawq_test_config, 'pattern_based'],
                          [create_hawq_hw_test_config, 'hw_config_vpu']))
def test_hawq_precision_init(_seed, dataset_dir, tmp_path, mocker, config_creator: Callable, filename_suffix: str):
    num_data_points = 100
    batch_size = 10
    config = config_creator(batch_size, num_data_points)
    model = MobileNetV2(num_classes=10)
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda()
    if not dataset_dir:
        dataset_dir = str(tmp_path)
    train_loader, _ = create_test_dataloaders(config.get("model_size"), dataset_dir, batch_size)
    config = register_default_init_args(config, criterion, train_loader)

    mocked_trace = mocker.patch('nncf.quantization.hessian_trace.HessianTraceEstimator.get_average_traces')

    mock_avg_traces = get_mock_avg_traces(model)
    mocked_trace.return_value = mock_avg_traces
    from torchvision.models.mobilenet import model_urls
    load_state(model, model_zoo.load_url(model_urls['mobilenet_v2']))
    model, algo_ctrl = create_compressed_model_and_algo_for_test(model, config)
    model = model.cuda()

    all_quantizers_per_full_scope = get_all_quantizers_per_full_scope(model)
    graph = get_bitwidth_graph(algo_ctrl, model, all_quantizers_per_full_scope)

    path_to_dot = 'mobilenet_v2_mixed_bitwidth_graph_{}.dot'.format(filename_suffix)
    check_graph(graph, path_to_dot, os.path.join('quantized', 'hawq'), sort_dot_graph=False)


def get_mock_avg_traces(model):
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


def get_bitwidth_graph(algo_ctrl, model, all_quantizers_per_full_scope):
    nncf_graph = model.get_graph()
    for node_key in nncf_graph.get_all_node_keys():
        node = nncf_graph.get_nx_node_by_key(node_key)
        node_id = node[NNCFGraph.ID_NODE_ATTR]
        color = ''
        if node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]:
            operator_name = node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].operator_name
            scope = node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic.scope_in_model
            module = model.get_module_by_scope(scope)
            if isinstance(module, NNCFConv2d):
                color = 'blue'
                if module.groups == module.in_channels:
                    operator_name = 'DW_Conv2d'
                    color = 'purple'

            node['label'] = '_#'.join([operator_name, str(node_id)])
            if color:
                node['color'] = color

    non_weight_quantizers = algo_ctrl.non_weight_quantizers
    bits_color_map = {4: 'red', 8: 'green', 6: 'orange'}
    for quantizer_id in non_weight_quantizers:
        activation_iap_ctx = quantizer_id.ia_op_exec_context
        post_hooked_nx_node_key = nncf_graph.get_node_id_by_iap_context(activation_iap_ctx)
        post_hooked_module_node = nncf_graph.get_nx_node_by_key(post_hooked_nx_node_key)
        operator_name = post_hooked_module_node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].operator_name
        node_id = post_hooked_module_node[NNCFGraph.ID_NODE_ATTR]
        post_hooked_module_node['label'] = '_#'.join([operator_name, str(node_id)])

        for next_nx_node_key in nncf_graph.get_successors(post_hooked_nx_node_key):
            activation_fq_node = nncf_graph.get_nx_node_by_key(next_nx_node_key)
            bits = non_weight_quantizers[quantizer_id].num_bits

            activation_fq_node['color'] = bits_color_map[bits]
            node_id = activation_fq_node[NNCFGraph.ID_NODE_ATTR]
            activation_fq_node['label'] = '{}_bit__AFQ_#{}'.format(bits, str(node_id))

    for scope, quantizer in all_quantizers_per_full_scope.items():
        if quantizer.is_weights:
            node = nncf_graph.find_node_in_nx_graph_by_scope(scope)
            if not node:
                raise AttributeError('Failed to get node by scope={}'.format(str(scope)))
            if node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]:
                bits = quantizer.num_bits
                node_id = node[NNCFGraph.ID_NODE_ATTR]
                node['label'] = '{}_bit__WFQ_#{}'.format(bits, str(node_id))
                node['color'] = bits_color_map[bits]
    return nncf_graph


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

    actual_configs = HAWQPrecisionInitializer.filter_configs_by_precision_constraints(
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


def test_disable_quantizer_gradients():
    config = get_basic_quantization_config()
    config['input_info'] = {
        "sample_size": [1, 3, 10, 10],
    }
    model = MobileNetV2(num_classes=10)
    model.eval()
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
    all_quantizations = get_all_modules_by_type(model, quantization_types)

    HAWQPrecisionInitializer.disable_quantizer_gradients(
        all_quantizations,
        compression_ctrl.quantized_weight_modules_registry,
        model)
    actual_state = get_requires_grad_per_param(model)
    path_to_ref = str(TEST_ROOT / 'data/hawq_reference/mobilenet_v2_requires_grad_per_param.json')
    compare_with_ref_if_exists(actual_state, path_to_ref)


def test_enable_quantizer_gradients():
    config = get_basic_quantization_config()
    config['input_info'] = {
        "sample_size": [1, 3, 10, 10],
    }
    model = MobileNetV2(num_classes=10)
    model.eval()
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
    all_quantizations = get_all_modules_by_type(model, quantization_types)

    original = get_requires_grad_per_param(model)
    disabled = HAWQPrecisionInitializer.disable_quantizer_gradients(
        all_quantizations,
        compression_ctrl.quantized_weight_modules_registry,
        model)
    HAWQPrecisionInitializer.enable_quantizer_gradients(model, all_quantizations, disabled)
    actual = get_requires_grad_per_param(model)
    assert original == actual


def get_path_to_bitwidth_dump(tmp_path, rank):
    out_file_path = tmp_path / 'bitwidth_per_scope_gpu{}.pt'.format(rank)
    return out_file_path


def hawq_dumping_worker(gpu, ngpus_per_node, config, tmp_path):
    config.batch_size = 3
    config.workers = 3
    config.gpu = gpu
    config.ngpus_per_node = ngpus_per_node
    config.rank = gpu
    config.distributed = True

    torch.distributed.init_process_group(backend="nccl", init_method='tcp://127.0.0.1:8899',
                                         world_size=config.world_size, rank=config.rank)

    model = safe_thread_call(partial(mobilenet_v2, pretrained=True))
    model.eval()

    input_infos_list = create_input_infos(config)
    input_sample_size = input_infos_list[0].shape
    data_loader = torch.utils.data.DataLoader(RankDatasetMock(input_sample_size[1:], config.rank),
                                              batch_size=3,
                                              num_workers=1,
                                              shuffle=False)
    criterion = torch.nn.MSELoss().cuda(config.gpu)
    config = register_default_init_args(config, criterion, data_loader)
    quant_model, compression_algo = create_compressed_model_and_algo_for_test(model, config)

    torch.cuda.set_device(config.gpu)
    quant_model.cuda(config.gpu)
    config.batch_size = int(config.batch_size / ngpus_per_node)
    config.workers = int(config.workers / ngpus_per_node)
    quant_model = torch.nn.parallel.DistributedDataParallel(quant_model, device_ids=[config.gpu])

    compression_algo.distributed()

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
