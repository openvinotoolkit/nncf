"""
 Copyright (c) 2021 Intel Corporation
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
import math
from collections import OrderedDict
from collections import namedtuple
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple

import os
import pytest
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial
from torch.utils import model_zoo
from torchvision.models import MobileNetV2
from torchvision.models import inception_v3
from torchvision.models import mobilenet_v2
from torchvision.models import resnet50
from torchvision.transforms import transforms

from examples.classification.main import create_cifar
from examples.common.sample_config import SampleConfig
from examples.object_detection.models.ssd_vgg import SSD_VGG
from nncf import register_default_init_args
from nncf.checkpoint_loading import load_state
from nncf.debug import set_debug_log_dir
from nncf.dynamic_graph.context import Scope
from nncf.dynamic_graph.graph_builder import create_input_infos
from nncf.hw_config import HWConfigType
from nncf.initialization import default_criterion_fn
from nncf.module_operations import UpdatePaddingValue
from nncf.quantization.adjust_padding import add_adjust_padding_nodes
from nncf.quantization.hessian_trace import HessianTraceEstimator
from nncf.quantization.layers import QUANTIZATION_MODULES
from nncf.quantization.layers import QuantizerConfig
from nncf.quantization.layers import QuantizersSwitcher
from nncf.quantization.precision_constraints import HardwareQuantizationConstraints
from nncf.quantization.precision_init.base_init import WeightQuantizersHandler
from nncf.quantization.precision_init.compression_ratio import CompressionRatioCalculator
from nncf.quantization.precision_init.hawq_debug import HAWQDebugger
from nncf.quantization.precision_init.hawq_init import BitwidthAssignmentMode
from nncf.quantization.precision_init.hawq_init import HAWQPrecisionInitializer
from nncf.quantization.precision_init.hawq_init import TraceOrderBitwidthMatcher
from nncf.quantization.precision_init.perturbations import PerturbationObserver
from nncf.quantization.precision_init.perturbations import Perturbations
from nncf.quantization.precision_init.traces_order import TracesOrder
from nncf.quantization.precision_init.traces_order import TracesPerLayer
from nncf.structures import QuantizationPrecisionInitArgs
from nncf.utils import get_all_modules_by_type
from nncf.utils import safe_thread_call
from tests.conftest import TEST_ROOT
from tests.helpers import BasicConvTestModel
from tests.helpers import create_compressed_model_and_algo_for_test
from tests.helpers import create_conv
from tests.quantization.test_quantization_helpers import compare_multi_gpu_dump
from tests.quantization.test_quantization_helpers import create_rank_dataloader
from tests.quantization.test_quantization_helpers import distributed_init_test_default
from tests.quantization.test_quantization_helpers import get_quantization_config_without_range_init
from tests.quantization.test_quantization_helpers import get_squeezenet_quantization_config
from tests.quantization.test_quantization_helpers import post_compression_test_distr_init
# pylint:disable=unused-import
from tests.modules.test_rnn import _seed
from tests.test_compressed_graph import check_nx_graph
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
        return self._set_target_device(HWConfigType.VPU.value)

    def for_cpu(self):
        return self._set_target_device(HWConfigType.CPU.value)

    def for_trial(self):
        return self._set_target_device('TRIAL')

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
    num_layers = len(get_all_modules_by_type(model, ['Conv2d', 'Linear']))
    return torch.randperm(num_layers).to(init_device) + 1


def check_bitwidth_graph(algo_ctrl, model, path_to_dot, graph_dir):
    if torch.cuda.is_available():
        model = model.cuda()
    all_quantizers_per_full_scope = HAWQDebugger.get_all_quantizers_per_full_scope(model)
    quantizer_switcher = QuantizersSwitcher(list(all_quantizers_per_full_scope.values()))
    # graph may not contain some quantizers (e.g. in staged scenario)
    quantizer_switcher.enable_quantizers()
    model.rebuild_graph()
    groups_of_adjacent_quantizers = algo_ctrl.groups_of_adjacent_quantizers
    graph = HAWQDebugger.get_bitwidth_graph(algo_ctrl, model,
                                            groups_of_adjacent_quantizers)
    nx_graph = add_adjust_padding_nodes(graph, model)
    check_nx_graph(nx_graph, path_to_dot, graph_dir, sort_dot_graph=False)


class HAWQTestStruct(NamedTuple):
    model_creator: Callable[[], nn.Module] = mobilenet_v2
    config_builder: HAWQConfigBuilder = HAWQConfigBuilder().for_vpu()
    filename_suffix: str = 'hw_config_vpu'
    avg_traces_creator: Callable[[nn.Module, str], torch.Tensor] = get_avg_traces

    def __str__(self):
        return '_'.join([self.model_creator.__name__, str(self.config_builder)])


HAWQ_TEST_PARAMS = (
    HAWQTestStruct(config_builder=HAWQConfigBuilder().staged()),
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
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    if not dataset_dir:
        dataset_dir = str(tmp_path)
    train_loader, _ = create_test_dataloaders(config, dataset_dir)
    config = register_default_init_args(config, train_loader, train_loader, criterion)

    mocked_trace = mocker.patch('nncf.quantization.hessian_trace.HessianTraceEstimator.get_average_traces',
                                autospec=True)
    pregen_traces_for_all_layers = params.avg_traces_creator(model, 'cuda')
    adjust_pad_creation_spy = mocker.spy(UpdatePaddingValue, '__init__')

    # There may be less traces required to be calculated during HAWQ than there are weightable layers.
    def side_effect_fn(self, max_iter=500, tolerance=1e-5):
        # pylint:disable=protected-access
        return pregen_traces_for_all_layers[:len(self._parameter_handler.parameters)]

    mocked_trace.side_effect = side_effect_fn
    model, algo_ctrl = create_compressed_model_and_algo_for_test(model, config)

    path_to_dot = '{}_{}.dot'.format(params.model_creator.__name__, params.config_builder.filename_suffix())
    graph_dir = os.path.join('quantized', 'hawq')
    check_bitwidth_graph(algo_ctrl, model, path_to_dot, graph_dir)


def test_hawq_hw_vpu_config_e2e(_seed, dataset_dir, tmp_path):
    config = HAWQConfigBuilder().for_vpu().liberal_mode().with_ratio(2.5).build()
    model = MobileNetV2(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    if not dataset_dir:
        dataset_dir = str(tmp_path)
    train_loader, _ = create_test_dataloaders(config, dataset_dir)
    config = register_default_init_args(config, train_loader, train_loader, criterion)

    create_compressed_model_and_algo_for_test(model, config)


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


def test_get_non_decreasing_bit_sequences():
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

    order = TracesOrder(list(range(L)))
    matcher = TraceOrderBitwidthMatcher(bits, order)
    actual_config = matcher.get_all_non_decreasing_bitwidth_sequences()
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
    assert len(parameters_to_restore.originally_disabled_gradients) == 406
    assert len(parameters_to_restore.skipped_gradients_to_enable) == 3
    actual_requires_grad_per_param = get_requires_grad_per_param(model)
    path_to_ref = str(TEST_ROOT / 'data/hawq_reference/mobilenet_v2_requires_grad_per_param.json')
    compare_with_ref_if_exists(actual_requires_grad_per_param, path_to_ref)


def test_enable_quantizer_gradients():
    switcher, params_to_restore, model, ctrl, origi_requires_grad_per_param = disable_quantizer_gradients()
    quantized_modules = ctrl.weight_quantizers
    HAWQPrecisionInitializer.restore_disabled_gradients(switcher, model, quantized_modules, params_to_restore)
    actual_requires_grad_per_param = get_requires_grad_per_param(model)
    assert origi_requires_grad_per_param == actual_requires_grad_per_param


def disable_quantizer_gradients():
    config = get_quantization_config_without_range_init()
    config['input_info'] = {
        "sample_size": [2, 3, 10, 10],
    }
    model = MobileNetV2(num_classes=10)
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    original_requires_grad_per_param = get_requires_grad_per_param(model)
    quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
    all_quantizations = get_all_modules_by_type(model, quantization_types)
    quantizers_switcher = QuantizersSwitcher(list(all_quantizations.values()))
    params_to_restore = HAWQPrecisionInitializer.disable_all_gradients_except_weights_of_quantized_modules(
        quantizers_switcher,
        compression_ctrl.weight_quantizers,
        model,
        get_scopes_of_skipped_weight_quantizers())
    return quantizers_switcher, params_to_restore, model, compression_ctrl, original_requires_grad_per_param


def get_path_to_bitwidth_dump(tmp_path, rank):
    out_file_path = tmp_path / 'bitwidth_per_scope_gpu{}.pt'.format(rank)
    return out_file_path


def precision_init_dumping_worker(gpu, ngpus_per_node, config, tmp_path):
    distributed_init_test_default(gpu, ngpus_per_node, config)
    data_loader = create_rank_dataloader(config, gpu)
    model = safe_thread_call(partial(mobilenet_v2, pretrained=True))
    model.eval()
    criterion = torch.nn.MSELoss().cuda(config.gpu)
    config = register_default_init_args(config, data_loader, None, criterion,
                                        autoq_eval_fn=lambda *x: 0, val_loader=data_loader)
    quant_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    quant_model = post_compression_test_distr_init(compression_ctrl, config, ngpus_per_node, quant_model)

    # just to reproduce the same scale values without Dropout
    quant_model.eval()

    act_bitwidth_per_scope = get_bitwidth_per_scope(quant_model.module)
    out_file_path = get_path_to_bitwidth_dump(tmp_path, config.rank)
    torch.save(act_bitwidth_per_scope, str(out_file_path))


def test_can_broadcast_initialized_precisions_in_distributed_mode(tmp_path, runs_subprocess_in_precommit):
    if not torch.cuda.is_available():
        pytest.skip("Skipping CUDA test cases for CPU only setups")
    config_builder = HAWQConfigBuilder(batch_size=2, num_data_points=10).for_trial()
    config = config_builder.build()
    ngpus_per_node = torch.cuda.device_count()
    config.world_size = ngpus_per_node
    torch.multiprocessing.spawn(precision_init_dumping_worker,
                                nprocs=ngpus_per_node,
                                args=(ngpus_per_node, config, tmp_path),
                                join=True)

    assert not compare_multi_gpu_dump(config, tmp_path, get_path_to_bitwidth_dump)


@pytest.mark.parametrize(('method_name', 'expected_behavior'),
                         [('_calc_traces', pytest.raises(RuntimeError))]
                         )
def test_hawq_behaviour__if_method_returns_none(mocker, method_name, expected_behavior):
    config = HAWQConfigBuilder().with_sample_size([1, 1, 4, 4]).for_trial().build()
    config['compression']['initializer']['range']['num_init_samples'] = 0
    model = BasicConvTestModel()
    mock_train_loader = mocker.stub()
    mock_train_loader.batch_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.register_extra_structs([QuantizationPrecisionInitArgs(criterion_fn=mocker.stub(),
                                                                 criterion=mocker.stub(),
                                                                 data_loader=mock_train_loader,
                                                                 device=device)])
    mocker.patch('nncf.quantization.algo.QuantizationController.run_batchnorm_adaptation')
    mocked_calc_traces = mocker.patch(
        'nncf.quantization.precision_init.hawq_init.HAWQPrecisionInitializer._calc_traces')
    stub = mocker.stub()
    stub.traces_order = TracesOrder([0])
    mocked_calc_traces.return_value = stub

    mocked_method = mocker.patch('nncf.quantization.precision_init.hawq_init.HAWQPrecisionInitializer.' + method_name)
    mocked_method.return_value = None

    with expected_behavior:
        create_compressed_model_and_algo_for_test(model, config)


def test_check_hawq_dump(mocker, tmp_path):
    tensor1 = torch.Tensor([1])
    tensor2 = torch.Tensor([2])
    qconf1 = QuantizerConfig(num_bits=2)
    qconf2 = QuantizerConfig(num_bits=4)
    id_ = 0
    quantizer_configurations = [[qconf1, qconf1], [qconf2, qconf2]]
    flops_per_config = [tensor1.item(), tensor2.item()]
    choosen_config_index = id_
    metric_per_qconfig_sequence = [tensor1, tensor2]
    perturbations = Perturbations()
    perturbations.add(id_, qconf1, tensor1)
    perturbations.add(id_, qconf2, tensor2)
    perturbations.add(id_ + 1, qconf1, tensor2)
    perturbations.add(id_ + 1, qconf2, tensor1)

    observer1 = PerturbationObserver(mocker.stub())
    observer1.perturbation = tensor1
    observer1.numels = id_
    observer1.input_norm = id_

    observer2 = PerturbationObserver(mocker.stub())
    observer2.perturbation = tensor2
    observer2.numels = id_
    observer2.input_norm = id_
    weight_observers = [observer1, observer2]
    traces_per_layer = TracesPerLayer(torch.cat((tensor1, tensor2)))

    set_debug_log_dir(str(tmp_path))
    hawq_debugger = HAWQDebugger(quantizer_configurations,
                                 perturbations,
                                 [weight_observers, weight_observers],
                                 traces_per_layer,
                                 [qconf1.num_bits, qconf2.num_bits])

    hawq_debugger.dump_metric_MB(metric_per_qconfig_sequence)
    hawq_debugger.dump_metric_flops(metric_per_qconfig_sequence, flops_per_config, choosen_config_index)
    hawq_debugger.dump_avg_traces()
    hawq_debugger.dump_density_of_quantization_noise()
    hawq_debugger.dump_perturbations_ratio()
    test_dir = tmp_path / Path('hawq_dumps')
    num_dump_files = len([name for name in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, name))])
    assert num_dump_files == 6


def get_quantization_config_with_ignored_scope():
    config = get_quantization_config_without_range_init()
    config['compression']['ignored_scopes'] = 'ConvLinear/NNCFLinear[fc]'
    return config


@pytest.mark.parametrize(('config_creator', 'ref_values'), (
        [
            get_quantization_config_without_range_init,
            (1.25, pytest.approx(1.42, abs=1e-2), (1, 2), (1, 4), (1, pytest.approx(1.8181, abs=1e-4)))
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

    handler = WeightQuantizersHandler(model, quantizers, HardwareQuantizationConstraints())
    ratio_calculator = CompressionRatioCalculator(model, handler)

    assert ratio_calculator.compression_ratio_for_bitwitdh_sequence([4, 8]) == ref_values[0]
    assert ratio_calculator.compression_ratio_for_bitwitdh_sequence([8, 4]) == ref_values[1]
    assert ratio_calculator.ratio_limits([4, 8]) == ref_values[2]
    assert ratio_calculator.ratio_limits([2, 4, 8]) == ref_values[3]
    constraints = HardwareQuantizationConstraints()
    constraints.add(list(quantizers)[0], {8})
    assert ratio_calculator.ratio_limits([2, 8], constraints) == ref_values[4]


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
    for quantizer_info in ctrl_load.weight_quantizers.values():
        assert quantizer_info.quantizer_module_ref.is_enabled_quantization()
