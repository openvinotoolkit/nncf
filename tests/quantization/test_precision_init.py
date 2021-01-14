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
from typing import Callable, NamedTuple, List, Dict

import math
import os
import pytest
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial
from random import random
from torch.utils import model_zoo
from torchvision.models import MobileNetV2, mobilenet_v2, resnet50, inception_v3
from torchvision.transforms import transforms

from examples.classification.main import create_cifar
from examples.common.model_loader import load_model
from examples.common.sample_config import SampleConfig
from examples.object_detection.models.ssd_vgg import SSD_VGG
from nncf import register_default_init_args, NNCFConfig
from nncf.checkpoint_loading import load_state
from nncf.debug import set_debug_log_dir
from nncf.dynamic_graph.context import Scope, ScopeElement
from nncf.dynamic_graph.graph_builder import create_input_infos
from nncf.hw_config import HWConfigType
from nncf.initialization import default_criterion_fn
from nncf.quantization.algo import QuantizerSetupType
from nncf.quantization.hessian_trace import HessianTraceEstimator
from nncf.quantization.hw_precision_constraints import HWPrecisionConstraints
from nncf.quantization.layers import QUANTIZATION_MODULES, QuantizerConfig, QuantizersSwitcher
from nncf.quantization.precision_init.adjacent_quantizers import GroupsOfAdjacentQuantizers
from nncf.quantization.precision_init.compression_ratio import CompressionRatioCalculator
from nncf.quantization.precision_init.hawq_debug import HAWQDebugger
from nncf.quantization.precision_init.hawq_init import BitwidthAssignmentMode, HAWQPrecisionInitializer
from nncf.quantization.precision_init.manual_init import WeightQuantizersHandler
from nncf.quantization.precision_init.perturbations import PerturbationObserver, Perturbations
from nncf.quantization.precision_init.traces_order import TracesOrder, TracesPerLayer
from nncf.quantization.quantizer_id import WeightQuantizerId
from nncf.utils import get_all_modules_by_type, safe_thread_call
from tests.conftest import TEST_ROOT, EXAMPLES_DIR
from tests.helpers import create_compressed_model_and_algo_for_test, create_conv, \
    create_mock_dataloader, BasicConvTestModel
from tests.quantization.test_quantization_helpers import compare_multi_gpu_dump, \
    get_quantization_config_without_range_init, distributed_init_test_default, post_compression_test_distr_init, \
    get_squeezenet_quantization_config, create_rank_dataloader
from tests.test_compressed_graph import check_graph

# pylint:disable=unused-import
from tests.modules.test_rnn import _seed
from tests.test_models import squeezenet1_1


def create_test_dataloaders(config, dataset_dir):
    input_info = create_input_infos(config)[0]
    image_size = input_info.shape[-1]
    batch_size = input_info.shape[0]
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))

    train_transforms = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    dummy_config = type('dummy', (object,), {'dataset_dir': dataset_dir})()
    train_dataset = create_cifar(dummy_config, dataset_config='cifar10', is_train=True, transform=train_transforms)

    # Do not set num_workers > 0 here - random hangs occur during pytest runs of this files
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                               pin_memory=True, drop_last=True)
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


class BaseConfigBuilder:
    def __init__(self, config_creator_fn: Callable = None):
        if config_creator_fn:
            self._config = config_creator_fn()
        self._options: Dict[str, str] = OrderedDict()
        self._extra_params: str = ''

    def with_ratio(self, ratio: float):
        self._config['compression']['initializer']['precision']['compression_ratio'] = ratio
        self._options['ratio'] = str(ratio)
        return self

    def _with_quantizer_setup_type(self, setup_type: QuantizerSetupType):
        self._config['quantizer_setup_type'] = setup_type.value
        self._options['setup_type'] = setup_type.value
        return self

    def prop_based(self):
        return self._with_quantizer_setup_type(QuantizerSetupType.PROPAGATION_BASED)

    def pattern_based(self):
        return self._with_quantizer_setup_type(QuantizerSetupType.PATTERN_BASED)

    def with_sample_size(self, sample_size: List[int]):
        self._config['input_info']['sample_size'] = sample_size
        return self

    def staged(self):
        self._config["compression"]["params"] = {
            "activations_quant_start_epoch": 0,
            "weights_quant_start_epoch": 1
        }
        self._extra_params += 'staged'
        return self

    def _set_target_device(self, config_type: str):
        self._config["target_device"] = config_type
        self._options['device'] = config_type
        return self

    def for_vpu(self):
        return self._set_target_device(HWConfigType.VPU.value).prop_based()

    def for_cpu(self):
        return self._set_target_device(HWConfigType.CPU.value).prop_based()

    def for_trial(self):
        return self._set_target_device('TRIAL').prop_based()

    def build(self):
        return self._config

    def with_ignored_scope(self, ignored_scopes=List[str]):
        self._config['ignored_scopes'] = ignored_scopes
        self._options['with'] = 'ignored_scopes'
        return self

    def __str__(self):
        if self._extra_params:
            return '_'.join([self.filename_suffix(), self._extra_params])
        return self.filename_suffix()

    def filename_suffix(self) -> str:
        ordered_options = OrderedDict(sorted(self._options.items()))
        return '__'.join(['_'.join([k, v]) for k, v in ordered_options.items()])


class HAWQConfigBuilder(BaseConfigBuilder):
    def __init__(self, config_creator_fn: Callable = None, batch_size=10, num_data_points=100, image_size=10):
        super().__init__(config_creator_fn)
        if not config_creator_fn:
            self._config = self.create_hawq_test_config(batch_size, num_data_points, image_size)
        self.num_data_points = num_data_points

    def _set_bitwidth_assignment_mode(self, mode: BitwidthAssignmentMode):
        self._config['compression']['initializer']['precision']['bitwidth_assignment_mode'] = mode.value
        self._options['mode'] = str(mode.value)
        return self

    def strict_mode(self):
        return self._set_bitwidth_assignment_mode(BitwidthAssignmentMode.STRICT)

    def liberal_mode(self):
        return self._set_bitwidth_assignment_mode(BitwidthAssignmentMode.LIBERAL)

    def build(self):
        return self._config

    def for_vpu(self):
        super().for_vpu()
        return self.strict_mode()

    @staticmethod
    def create_hawq_test_config(batch_size=10, num_data_points=100, image_size=10):
        config = get_quantization_config_without_range_init()
        config['input_info'] = {
            "sample_size": [batch_size, 3, image_size, image_size],
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
                    'num_init_samples': 1
                },
                'batchnorm_adaptation': {
                    'num_bn_adaptation_samples': 0,
                    'num_bn_forget_samples': 0
                }
            }})
        return config


def ssd_vgg_512_test():
    ssd_params = SampleConfig({
        "steps": [8, 16, 32, 64, 128, 256, 512],
        "min_sizes": [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],
        "max_sizes": [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
        "variance": [0.1, 0.1, 0.2, 0.2],
        "clip": False,
        "flip": True
    })
    return SSD_VGG(cfg=ssd_params, size=512, num_classes=21)


def get_avg_traces(model, init_device: str):
    num_traces = len(get_all_modules_by_type(model, ['Conv2d', 'Linear']))
    return torch.randperm(num_traces).to(init_device) + 1


def check_bitwidth_graph(algo_ctrl, model, path_to_dot, graph_dir):
    model = model.cuda()
    all_quantizers_per_full_scope = HAWQDebugger.get_all_quantizers_per_full_scope(model)
    quantizer_switcher = QuantizersSwitcher(list(all_quantizers_per_full_scope.values()))
    # graph may not contain some quantizers (e.g. in staged scenario)
    quantizer_switcher.enable_quantizers()
    model.rebuild_graph()
    groups_of_adjacent_quantizers = GroupsOfAdjacentQuantizers(algo_ctrl)
    graph = HAWQDebugger.get_bitwidth_graph(algo_ctrl, model, all_quantizers_per_full_scope,
                                            groups_of_adjacent_quantizers)
    check_graph(graph, path_to_dot, graph_dir, sort_dot_graph=False)


class HAWQTestStruct(NamedTuple):
    model_creator: Callable[[], nn.Module] = mobilenet_v2
    config_builder: HAWQConfigBuilder = HAWQConfigBuilder().prop_based().for_vpu()
    filename_suffix: str = 'hw_config_vpu'
    avg_traces_creator: Callable[[nn.Module, str], torch.Tensor] = get_avg_traces

    def __str__(self):
        return '_'.join([self.model_creator.__name__, str(self.config_builder)])


HAWQ_TEST_PARAMS = (
    HAWQTestStruct(config_builder=HAWQConfigBuilder().pattern_based()),
    HAWQTestStruct(config_builder=HAWQConfigBuilder().staged().pattern_based()),
    HAWQTestStruct(config_builder=HAWQConfigBuilder().for_trial()),
    HAWQTestStruct(config_builder=HAWQConfigBuilder().for_cpu()),
    HAWQTestStruct(config_builder=HAWQConfigBuilder().for_vpu().liberal_mode().with_ratio(2.5)),
    HAWQTestStruct(config_builder=HAWQConfigBuilder().with_ratio(1.02).for_vpu()),
    HAWQTestStruct(model_creator=squeezenet1_1,
                   config_builder=HAWQConfigBuilder().with_sample_size([1, 3, 224, 224]).for_vpu()),
    HAWQTestStruct(model_creator=resnet50,
                   config_builder=HAWQConfigBuilder().with_ratio(1.11).for_vpu()),
    HAWQTestStruct(model_creator=resnet50,
                   config_builder=HAWQConfigBuilder().for_vpu().liberal_mode().with_ratio(2.5)),
    HAWQTestStruct(model_creator=inception_v3,
                   avg_traces_creator=lambda x, y: get_avg_traces(x, y)[:95],
                   config_builder=HAWQConfigBuilder().with_sample_size([2, 3, 299, 299]).for_vpu().with_ratio(1)),
    HAWQTestStruct(model_creator=inception_v3,
                   avg_traces_creator=lambda x, y: get_avg_traces(x, y)[:94],
                   config_builder=HAWQConfigBuilder().with_sample_size([2, 3, 299, 299]).for_vpu().liberal_mode().
                   with_ignored_scope(['Inception3/BasicConv2d[Conv2d_2a_3x3]']).with_ratio(2.5)),
    HAWQTestStruct(model_creator=inception_v3,
                   avg_traces_creator=lambda x, y: get_avg_traces(x, y)[:95],
                   config_builder=HAWQConfigBuilder().with_sample_size(
                       [2, 3, 299, 299]).for_vpu().liberal_mode().with_ratio(2.5)),
    HAWQTestStruct(model_creator=ssd_vgg_512_test,
                   config_builder=HAWQConfigBuilder().with_sample_size([1, 3, 512, 512]).for_vpu().with_ratio(1.09)),
    HAWQTestStruct(model_creator=ssd_vgg_512_test,
                   config_builder=HAWQConfigBuilder().with_sample_size(
                       [1, 3, 512, 512]).for_vpu().liberal_mode().with_ratio(2.5)),
)


@pytest.mark.parametrize('params', HAWQ_TEST_PARAMS, ids=[str(p) for p in HAWQ_TEST_PARAMS])
def test_hawq_precision_init(_seed, dataset_dir, tmp_path, mocker, params):
    config = params.config_builder.build()
    model = params.model_creator()

    criterion = nn.CrossEntropyLoss().cuda()
    if not dataset_dir:
        dataset_dir = str(tmp_path)
    train_loader, _ = create_test_dataloaders(config, dataset_dir)
    config = register_default_init_args(config, train_loader, criterion)

    mocked_trace = mocker.patch('nncf.quantization.hessian_trace.HessianTraceEstimator.get_average_traces')
    mocked_trace.return_value = params.avg_traces_creator(model, 'cuda')
    model, algo_ctrl = create_compressed_model_and_algo_for_test(model, config)

    path_to_dot = '{}_{}.dot'.format(params.model_creator.__name__, params.config_builder.filename_suffix())
    graph_dir = os.path.join('quantized', 'hawq')
    check_bitwidth_graph(algo_ctrl, model, path_to_dot, graph_dir)


class AutoQConfigBuilder(BaseConfigBuilder):
    def __init__(self, config_creator_fn: Callable = None, batch_size=10, image_size=10, num_channels=3,
                 num_init_samples=1):
        super().__init__(config_creator_fn)
        if not config_creator_fn:
            self._config = self.create_autoq_test_config(batch_size, image_size, num_channels,
                                                         num_init_samples=num_init_samples)
        self.for_vpu()

    def eval_subset_ratio(self, eval_subset_ratio):
        self._options['eval_subset_ratio'] = str(eval_subset_ratio)
        self._config['compression']['initializer']['precision']['eval_subset_ratio'] = eval_subset_ratio
        return self

    def iter_number(self, iter_number):
        self._options['iter_number'] = str(iter_number)
        self._config['compression']['initializer']['precision']['iter_number'] = iter_number
        return self

    def warmup_iter_number(self, warmup_iter_number):
        self._options['warmup_iter_number'] = str(warmup_iter_number)
        self._config['compression']['initializer']['precision']['warmup_iter_number'] = warmup_iter_number
        return self

    @staticmethod
    def create_autoq_test_config(batch_size=10, image_size=10, num_channels=3, num_init_samples=1):
        config = get_quantization_config_without_range_init()
        config['input_info'] = {
            "sample_size": [batch_size, num_channels, image_size, image_size],
        }
        config['batch_size'] = batch_size
        config['compression'].update({
            'initializer': {
                'precision': {
                    "type": "autoq",
                    "bits": [2, 4, 8],
                    "iter_number": 2,
                    "compression_ratio": 0.15,
                    "eval_subset_ratio": 1.0,
                    "warmup_iter_number": 1
                },
                'range': {
                    'num_init_samples': num_init_samples
                },
                'batchnorm_adaptation': {
                    'num_bn_adaptation_samples': 0,
                    'num_bn_forget_samples': 0
                }
            }})
        return config


class AutoQTestStruct(NamedTuple):
    model_creator: Callable[[], nn.Module] = mobilenet_v2
    config_builder: AutoQConfigBuilder = AutoQConfigBuilder().for_vpu()
    filename_suffix: str = 'hw_config_vpu'

    def __str__(self):
        return '_'.join([self.model_creator.__name__, str(self.config_builder)])


RATIO = 0.4
AUTOQ_TEST_PARAMS = (
    AutoQTestStruct(config_builder=AutoQConfigBuilder()),
    AutoQTestStruct(config_builder=AutoQConfigBuilder().with_ratio(RATIO)),
    AutoQTestStruct(config_builder=AutoQConfigBuilder().with_ratio(RATIO).eval_subset_ratio(RATIO)),
    AutoQTestStruct(config_builder=AutoQConfigBuilder().eval_subset_ratio(RATIO)),
    AutoQTestStruct(model_creator=squeezenet1_1,
                    config_builder=AutoQConfigBuilder().with_sample_size([1, 3, 224, 224])),
    AutoQTestStruct(model_creator=resnet50,
                    config_builder=AutoQConfigBuilder()),
    AutoQTestStruct(model_creator=resnet50,
                    config_builder=AutoQConfigBuilder().iter_number(4).warmup_iter_number(2)),
    AutoQTestStruct(model_creator=resnet50,
                    config_builder=AutoQConfigBuilder().with_ratio(RATIO)),
    AutoQTestStruct(model_creator=resnet50,
                    config_builder=AutoQConfigBuilder().eval_subset_ratio(RATIO)),
    AutoQTestStruct(model_creator=resnet50,
                    config_builder=AutoQConfigBuilder().with_ratio(RATIO).eval_subset_ratio(RATIO)),
    AutoQTestStruct(model_creator=inception_v3,
                    config_builder=AutoQConfigBuilder().with_sample_size([2, 3, 299, 299]).with_ratio(RATIO)),
    AutoQTestStruct(model_creator=inception_v3,
                    config_builder=AutoQConfigBuilder().with_sample_size([2, 3, 299, 299]).
                    with_ignored_scope(['Inception3/BasicConv2d[Conv2d_2a_3x3]']).eval_subset_ratio(RATIO)),
    AutoQTestStruct(model_creator=ssd_vgg_512_test,
                    config_builder=AutoQConfigBuilder().with_sample_size([1, 3, 512, 512]).eval_subset_ratio(RATIO)),
    AutoQTestStruct(model_creator=ssd_vgg_512_test,
                    config_builder=AutoQConfigBuilder().with_sample_size([1, 3, 512, 512]).with_ratio(RATIO)),
)


@pytest.mark.parametrize('params', AUTOQ_TEST_PARAMS, ids=[str(p) for p in AUTOQ_TEST_PARAMS])
def test_autoq_precision_init(_seed, dataset_dir, tmp_path, mocker, params):
    config = params.config_builder.build()
    model = params.model_creator()
    config['log_dir'] = str(tmp_path)

    if not dataset_dir:
        dataset_dir = str(tmp_path)
    train_loader, _ = create_test_dataloaders(config, dataset_dir)

    from nncf.automl.agent.ddpg.ddpg import DDPG
    random_action_spy = mocker.spy(DDPG, 'random_action')
    select_action_spy = mocker.spy(DDPG, 'select_action')

    config = register_default_init_args(config, train_loader=train_loader,
                                        autoq_eval_fn=lambda *x: random(),
                                        autoq_eval_loader=train_loader)
    model, algo_ctrl = create_compressed_model_and_algo_for_test(model, config)

    bw_init_config = config['compression']['initializer']['precision']
    learning_iter_number = bw_init_config['iter_number'] - bw_init_config['warmup_iter_number']
    n_quantizer = len(algo_ctrl.all_quantizations)

    assert random_action_spy.call_count == bw_init_config['warmup_iter_number'] * n_quantizer
    assert select_action_spy.call_count == learning_iter_number * (n_quantizer+1) + bw_init_config['warmup_iter_number']

    path_to_dot = '{}_{}.dot'.format(params.model_creator.__name__, params.config_builder.filename_suffix())
    graph_dir = os.path.join('quantized', 'autoq')
    check_bitwidth_graph(algo_ctrl, model, path_to_dot, graph_dir)


def test_hawq_hw_vpu_config_e2e(_seed, dataset_dir, tmp_path):
    config = HAWQConfigBuilder().for_vpu().liberal_mode().with_ratio(2.5).build()
    model = MobileNetV2(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    if not dataset_dir:
        dataset_dir = str(tmp_path)
    train_loader, _ = create_test_dataloaders(config, dataset_dir)
    config = register_default_init_args(config, train_loader, criterion)

    create_compressed_model_and_algo_for_test(model, config)


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
    traces_order = TracesOrder(order)
    ref_configurations = [bits_configurations[config_id] for config_id in survived_configurations_id]
    ordered_weight_keys = [get_mock_quantizer_id(str(i)) for i in range(len(constraints))]
    hw_precision_constraints = get_mock_precision_constraints(constraints, ordered_weight_keys)

    # pylint:disable=protected-access
    actual_configs = HAWQPrecisionInitializer._filter_configs_by_precision_constraints(
        bits_configurations, hw_precision_constraints, ordered_weight_keys, traces_order)

    assert ref_configurations == actual_configs


HAWQTestParams = namedtuple('HAWQTestParams', ('iter_number', 'batch_size', 'num_data_points', 'ref_trace'))


@pytest.mark.parametrize("params",
                         (HAWQTestParams(200, 13, 100, 0.04771214351058006),
                          HAWQTestParams(2, 13, 100, 0.031417448073625565),
                          HAWQTestParams(2, 10, 10, 0.04505229741334915),
                          HAWQTestParams(2, 10, 5, 0.04505229741334915)),
                         ids=('until_threshold', 'until_num_iter', 'batch_eq_num_data', 'batch_larger_num_data'))
def test_hawq_on_single_conv_without_quantizers(_seed, dataset_dir, tmp_path, params: HAWQTestParams):
    config = get_squeezenet_quantization_config(batch_size=params.batch_size)
    iter_number = params.iter_number
    tolerance = 4e-4

    model = squeezenet1_1(num_classes=10, dropout=0)
    from torchvision.models.squeezenet import model_urls
    load_state(model, model_zoo.load_url(model_urls['squeezenet1_1']))
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    if not dataset_dir:
        dataset_dir = str(tmp_path)
    data_loader, _ = create_test_dataloaders(config, dataset_dir)
    device = next(model.parameters()).device

    for _, param in model.named_parameters():
        param.requires_grad = False
    first_conv = next(iter(get_all_modules_by_type(model, 'Conv2d').values()))
    first_conv.weight.requires_grad = True

    trace_estimator = HessianTraceEstimator(model, default_criterion_fn, criterion, device, data_loader,
                                            params.num_data_points)
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
    actual_config = HAWQPrecisionInitializer.get_configs_constrained_by_traces_order(bits, L)
    ref_num = get_size_of_search_space(m, L)
    assert len(ref_configs) == ref_num
    assert len(actual_config) == ref_num
    assert sorted(actual_config) == sorted(ref_configs)


def get_requires_grad_per_param(model):
    not_sorted = OrderedDict({param_name: param.requires_grad for param_name, param in model.named_parameters()})
    return OrderedDict(sorted(not_sorted.items()))


def get_scopes_of_skipped_weight_quantizers():
    scopes_list = ['MobileNetV2/Sequential[features]/ConvBNReLU[18]/NNCFConv2d[0]',
                   'MobileNetV2/Sequential[features]/InvertedResidual[17]/Sequential[conv]/NNCFConv2d[2]',
                   'MobileNetV2/Sequential[features]/InvertedResidual[16]/Sequential[conv]/NNCFConv2d[2]']
    return [Scope.from_str(s) for s in scopes_list]


def test_disable_quantizer_gradients():
    _, parameters_to_restore, model, *_ = disable_quantizer_gradients()
    assert len(parameters_to_restore.originally_disabled_gradients) == 354
    assert len(parameters_to_restore.skipped_gradients_to_enable) == 3
    actual_requires_grad_per_param = get_requires_grad_per_param(model)
    path_to_ref = str(TEST_ROOT / 'data/hawq_reference/mobilenet_v2_requires_grad_per_param.json')
    compare_with_ref_if_exists(actual_requires_grad_per_param, path_to_ref)


def test_enable_quantizer_gradients():
    switcher, params_to_restore, model, ctrl, origi_requires_grad_per_param = disable_quantizer_gradients()
    quantized_modules = ctrl.quantized_weight_modules_registry
    HAWQPrecisionInitializer.restore_disabled_gradients(switcher, model, quantized_modules, params_to_restore)
    actual_requires_grad_per_param = get_requires_grad_per_param(model)
    assert origi_requires_grad_per_param == actual_requires_grad_per_param


def disable_quantizer_gradients():
    config = get_quantization_config_without_range_init()
    config['input_info'] = {
        "sample_size": [2, 3, 10, 10],
    }
    config['quantizer_setup_type'] = 'pattern_based'
    model = MobileNetV2(num_classes=10)
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    original_requires_grad_per_param = get_requires_grad_per_param(model)
    quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
    all_quantizations = get_all_modules_by_type(model, quantization_types)
    quantizers_switcher = QuantizersSwitcher(list(all_quantizations.values()))
    params_to_restore = HAWQPrecisionInitializer.disable_all_gradients_except_weights_of_quantized_modules(
        quantizers_switcher,
        compression_ctrl.quantized_weight_modules_registry,
        model,
        get_scopes_of_skipped_weight_quantizers())
    return quantizers_switcher, params_to_restore, model, compression_ctrl, original_requires_grad_per_param


def get_path_to_bitwidth_dump(tmp_path, rank):
    out_file_path = tmp_path / 'bitwidth_per_scope_gpu{}.pt'.format(rank)
    return out_file_path


def hawq_dumping_worker(gpu, ngpus_per_node, config, tmp_path):
    distributed_init_test_default(gpu, ngpus_per_node, config)
    data_loader = create_rank_dataloader(config, gpu)
    model = safe_thread_call(partial(mobilenet_v2, pretrained=True))
    model.eval()
    criterion = torch.nn.MSELoss().cuda(config.gpu)
    config = register_default_init_args(config, data_loader, criterion,
                                        autoq_eval_fn=lambda *x: 0, autoq_eval_loader=data_loader)
    quant_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    quant_model = post_compression_test_distr_init(compression_ctrl, config, ngpus_per_node, quant_model)

    # just to reproduce the same scale values without Dropout
    quant_model.eval()

    act_bitwidth_per_scope = get_bitwidth_per_scope(quant_model.module)
    out_file_path = get_path_to_bitwidth_dump(tmp_path, config.rank)
    torch.save(act_bitwidth_per_scope, str(out_file_path))


@pytest.mark.parametrize('config_builder', [HAWQConfigBuilder(batch_size=2, num_data_points=10).for_trial(),
                                            AutoQConfigBuilder(batch_size=2).for_trial()])
def test_can_broadcast_initialized_precisions_in_distributed_mode(config_builder, tmp_path):
    config = config_builder.build()
    ngpus_per_node = torch.cuda.device_count()
    config.world_size = ngpus_per_node
    torch.multiprocessing.spawn(hawq_dumping_worker,
                                nprocs=ngpus_per_node,
                                args=(ngpus_per_node, config, tmp_path),
                                join=True)

    assert not compare_multi_gpu_dump(config, tmp_path, get_path_to_bitwidth_dump)


ManualConfigTestParams = namedtuple('ManualConfigTestParams', ('config_name', 'bit_stats'))
MANUAL_CONFIG_TEST_PARAMS = [
    ManualConfigTestParams(config_name="mobilenet_v2_imagenet_mixed_int_manual.json",
                           bit_stats=[['8', '23.077', '23.932', '47.009'],
                                      ['4', '22.222', '30.769', '52.991']]),
    ManualConfigTestParams(config_name="resnet50_imagenet_mixed_int_manual.json",
                           bit_stats=[['8', '21.600', '23.200', '44.800'],
                                      ['4', '21.600', '33.600', '55.200']]),
    ManualConfigTestParams(config_name="squeezenet1_1_imagenet_mixed_int_manual.json",
                           bit_stats=[['8', '24.528', '30.189', '54.717'],
                                      ['4', '24.528', '20.755', '45.283']]),
    ManualConfigTestParams(config_name="squeezenet1_1_imagenet_mixed_int_manual_staged.json",
                           bit_stats=[['8', '24.528', '30.189', '54.717'],
                                      ['4', '24.528', '20.755', '45.283']])
]


@pytest.mark.parametrize('manual_config_params', MANUAL_CONFIG_TEST_PARAMS,
                         ids=[pair[0] for pair in MANUAL_CONFIG_TEST_PARAMS])
def test_hawq_manual_configs(manual_config_params):
    config_name, bit_stats = manual_config_params
    config_path = EXAMPLES_DIR.joinpath('classification', 'configs', 'mixed_precision') / config_name
    config = NNCFConfig.from_json(str(config_path))
    config['quantizer_setup_type'] = 'pattern_based'
    config = register_default_init_args(config, train_loader=create_mock_dataloader(config), criterion=None)
    model = load_model(config['model'], pretrained=False)
    model.eval()

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    table = compression_ctrl.non_stable_metric_collectors[0].get_bits_stat()
    # pylint: disable=protected-access
    assert table._rows == bit_stats


@pytest.mark.parametrize(('method_name', 'expected_behavior'),
                         [
                             ('_calc_traces', pytest.raises(RuntimeError)),
                             ('_filter_configs_by_precision_constraints', pytest.warns(RuntimeWarning))]
                         )
def test_hawq_behaviour__if_method_returns_none(mocker, method_name, expected_behavior):
    config = HAWQConfigBuilder().with_sample_size([1, 1, 4, 4]).build()
    config['quantizer_setup_type'] = 'pattern_based'
    model = BasicConvTestModel()
    mock_train_loader = mocker.stub()
    mock_train_loader.batch_size = 1
    config = register_default_init_args(config, mock_train_loader, mocker.stub())
    mocker.patch('nncf.quantization.algo.QuantizationController.run_batchnorm_adaptation')
    mocker.patch('nncf.quantization.algo.QuantizationController._do_range_init')
    mocker.patch('nncf.quantization.precision_init.hawq_init.HAWQPrecisionInitializer._calc_traces')

    mocked_trace = mocker.patch('nncf.quantization.precision_init.hawq_init.HAWQPrecisionInitializer.' + method_name)
    mocked_trace.return_value = None

    with expected_behavior:
        create_compressed_model_and_algo_for_test(model, config)


def test_check_hawq_dump(mocker, tmp_path):
    tensor1 = torch.Tensor([1])
    tensor2 = torch.Tensor([2])
    bitwidth1 = 2
    bitwidth2 = 4
    id_ = 0
    bits_configurations = [[bitwidth1], [bitwidth2]]
    flops_per_config = [tensor1.item(), tensor2.item()]
    choosen_config_index = id_
    configuration_metric = [tensor1, tensor2]
    perturbations = Perturbations()
    perturbations.add(id_, bitwidth1, tensor1)
    perturbations.add(id_, bitwidth2, tensor2)
    observer = PerturbationObserver(mocker.stub())
    observer.perturbation = tensor1
    observer.numels = id_
    observer.input_norm = id_
    weight_observers = [observer]
    traces_per_layer = TracesPerLayer(torch.cat((tensor1, tensor2)))

    set_debug_log_dir(str(tmp_path))
    hawq_debugger = HAWQDebugger(bits_configurations, perturbations, weight_observers, traces_per_layer,
                                 [bitwidth1, bitwidth2])

    hawq_debugger.dump_metric_MB(configuration_metric)
    hawq_debugger.dump_metric_flops(configuration_metric, flops_per_config, choosen_config_index)
    hawq_debugger.dump_avg_traces()
    hawq_debugger.dump_density_of_quantization_noise()
    hawq_debugger.dump_perturbations_ratio()
    test_dir = tmp_path / Path('hawq_dumps')
    num_dump_files = len([name for name in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, name))])
    assert num_dump_files == 6


#       fq_2
#        \
# fq_2 - conv_1 - fq_6
#                   \
#        fq_4       add
#         \         /
# fq_4 - conv_2 - fq_6
#
class ModelForTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 2, 2, -1, -2)
        self.conv2 = create_conv(1, 2, 2, -1, -2)

    def forward(self, x):
        return self.conv1(x) + self.conv2(x)


def test_quantization_configs__with_precisions_list():
    model = ModelForTest()

    config = get_quantization_config_without_range_init()
    config['compression']['initializer'].update({
        "precision": {
            "bitwidth_per_scope":
                [[2, 'ModelForTest/NNCFConv2d[conv1]'],
                 [4, 'ModelForTest/NNCFConv2d[conv2]']]
        }})
    config['compression']["activations"] = {"bits": 6}
    config['quantizer_setup_type'] = 'pattern_based'
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
    table = compression_ctrl.non_stable_metric_collectors[0].get_bits_stat()
    # pylint: disable=protected-access
    assert table._rows == ref_rows


def get_quantization_config_with_ignored_scope():
    config = get_quantization_config_without_range_init()
    config['compression']['ignored_scopes'] = 'ConvLinear/NNCFLinear[fc]'
    return config


@pytest.mark.parametrize(('config_creator', 'ref_values'), (
    [
        get_quantization_config_without_range_init,
        (1.75, pytest.approx(1.07, abs=1e-2), (1, 2), (1, 4), (1, 1.12))
    ],
    [
        get_quantization_config_with_ignored_scope,
        (2, 1, (1, 2), (1, 4), (1, 1))
    ]
))
def test_flops(config_creator, ref_values):
    class ConvLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = create_conv(1, 1, 2, -1, -2)
            self.fc = nn.Linear(3, 6)

        def forward(self, x):
            return self.fc(self.conv1(x))

    config = config_creator()
    model, compression_ctrl = create_compressed_model_and_algo_for_test(ConvLinear(), config)
    quantizers = compression_ctrl.weight_quantizers

    handler = WeightQuantizersHandler(model, quantizers, HWPrecisionConstraints(True))
    flops_counter = CompressionRatioCalculator(model, handler)

    assert flops_counter.ratio_for_bits_configuration([4, 8]) == ref_values[0]
    assert flops_counter.ratio_for_bits_configuration([8, 4]) == ref_values[1]
    assert flops_counter.ratio_limits([4, 8]) == ref_values[2]
    assert flops_counter.ratio_limits([2, 4, 8]) == ref_values[3]
    constraints = HWPrecisionConstraints(True).add(list(quantizers)[0], [QuantizerConfig(bits=8)])
    assert flops_counter.ratio_limits([2, 8], constraints) == ref_values[4]


def test_staged_quantization_saves_enabled_quantizers_in_state_dict(tmp_path):
    config = get_quantization_config_without_range_init()
    config["compression"]["params"] = {
        "activations_quant_start_epoch": 2,
        "weights_quant_start_epoch": 1
    }
    model_save, ctrl_save = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config)
    ctrl_save.scheduler.epoch_step()
    ctrl_save.scheduler.epoch_step()
    _, ctrl_load = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config,
                                                             resuming_state_dict=model_save.state_dict())
    for quantizer_info in ctrl_load.non_weight_quantizers.values():
        assert not quantizer_info.quantizer_module_ref.is_enabled_quantization()
    for quantizer in ctrl_load.weight_quantizers.values():
        assert quantizer.is_enabled_quantization()
