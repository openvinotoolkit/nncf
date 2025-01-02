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
import itertools
import json
import math
import os
from collections import OrderedDict
from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple

import pytest
import torch
import torch.utils.data
from numpy.random import random_sample
from torch import nn
from torchvision.models import resnet50
from torchvision.transforms import transforms

import nncf
from examples.common.sample_config import SampleConfig
from examples.torch.classification.main import create_cifar
from examples.torch.object_detection.models.ssd_vgg import SSD_VGG
from nncf import NNCFConfig
from nncf.common.graph import NNCFNodeName
from nncf.common.hardware.config import HWConfigType
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.utils.debug import set_debug_log_dir
from nncf.torch import register_default_init_args
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.dynamic_graph.io_handling import FillerInputInfo
from nncf.torch.initialization import default_criterion_fn
from nncf.torch.quantization.adjust_padding import add_adjust_padding_nodes
from nncf.torch.quantization.hessian_trace import HessianTraceEstimator
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import QuantizerConfig
from nncf.torch.quantization.layers import QuantizersSwitcher
from nncf.torch.quantization.precision_init.bitwidth_graph import BitwidthGraph
from nncf.torch.quantization.precision_init.compression_ratio import CompressionRatioCalculator
from nncf.torch.quantization.precision_init.hawq_debug import HAWQDebugger
from nncf.torch.quantization.precision_init.hawq_init import BitwidthAssignmentMode
from nncf.torch.quantization.precision_init.hawq_init import HAWQPrecisionInitializer
from nncf.torch.quantization.precision_init.hawq_init import TraceOrderBitwidthMatcher
from nncf.torch.quantization.precision_init.perturbations import PerturbationObserver
from nncf.torch.quantization.precision_init.perturbations import Perturbations
from nncf.torch.quantization.precision_init.traces_order import TracesOrder
from nncf.torch.quantization.precision_init.traces_order import TracesPerLayer
from nncf.torch.structures import QuantizationPrecisionInitArgs
from nncf.torch.utils import get_all_modules_by_type
from nncf.torch.utils import get_model_device
from nncf.torch.utils import safe_thread_call
from tests.cross_fw.shared.nx_graph import compare_nx_graph_with_reference
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_conv
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.quantization.quantization_helpers import compare_multi_gpu_dump
from tests.torch.quantization.quantization_helpers import create_rank_dataloader
from tests.torch.quantization.quantization_helpers import distributed_init_test_default
from tests.torch.quantization.quantization_helpers import get_quantization_config_without_range_init
from tests.torch.quantization.quantization_helpers import get_squeezenet_quantization_config
from tests.torch.quantization.quantization_helpers import post_compression_test_distr_init
from tests.torch.test_compressed_graph import get_full_path_to_the_graph
from tests.torch.test_models import inception_v3
from tests.torch.test_models import squeezenet1_1
from tests.torch.test_models.mobilenet import MobileNetV2
from tests.torch.test_models.mobilenet import mobilenet_v2


def create_test_dataloaders(config: NNCFConfig, dataset_dir):
    input_info = FillerInputInfo.from_nncf_config(config).elements[0]
    image_size = input_info.shape[-1]
    batch_size = input_info.shape[0]
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dummy_config = type("dummy", (object,), {"dataset_dir": dataset_dir})()
    train_dataset = create_cifar(dummy_config, dataset_config="cifar10", is_train=True, transform=train_transforms)

    # Do not set num_workers > 0 here - random hangs occur during pytest runs of this files
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True
    )
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
        with open(path_to_ref, "r", encoding="utf8") as f:
            assert json.load(f) == actual_state
    else:
        with open(path_to_ref, "w", encoding="utf8") as f:
            json.dump(actual_state, f)


class BaseConfigBuilder:
    def __init__(self, config_creator_fn: Callable = None):
        if config_creator_fn:
            self._config = config_creator_fn()
        self._options: Dict[str, str] = OrderedDict()
        self._extra_params: str = ""

    def with_ratio(self, ratio: float):
        self._config["compression"]["initializer"]["precision"]["compression_ratio"] = ratio
        self._options["ratio"] = str(ratio)
        return self

    def with_sample_size(self, sample_size: List[int]):
        self._config["input_info"]["sample_size"] = sample_size
        return self

    def staged(self):
        self._config["compression"]["params"] = {"activations_quant_start_epoch": 0, "weights_quant_start_epoch": 1}
        self._extra_params += "staged"
        return self

    def _set_target_device(self, config_type: str):
        self._config["target_device"] = config_type
        self._options["device"] = config_type
        return self

    def for_npu(self):
        return self._set_target_device(HWConfigType.NPU.value)

    def for_cpu(self):
        return self._set_target_device(HWConfigType.CPU.value)

    def for_trial(self):
        return self._set_target_device("TRIAL")

    def build(self):
        return self._config

    def with_ignored_scope(self, ignored_scopes=List[str], target_group: QuantizerGroup = None):
        if target_group is None:
            self._config["compression"]["ignored_scopes"] = ignored_scopes
        else:
            if target_group.value not in self._config["compression"]:
                self._config["compression"][target_group.value] = {}
            self._config["compression"][target_group.value]["ignored_scopes"] = ignored_scopes
        self._options["with"] = "ignored_scopes"
        return self

    def with_target_scope(self, target_scopes=List[str]):
        self._config["target_scopes"] = target_scopes
        self._config["compression"]["target_scopes"] = target_scopes
        self._options["with"] = "target_scopes"
        return self

    def __str__(self):
        if self._extra_params:
            return "_".join([self.filename_suffix(), self._extra_params])
        return self.filename_suffix()

    def filename_suffix(self) -> str:
        ordered_options = OrderedDict(sorted(self._options.items()))
        return "__".join(["_".join([k, v]) for k, v in ordered_options.items()])


class HAWQConfigBuilder(BaseConfigBuilder):
    def __init__(self, config_creator_fn: Callable = None, batch_size=10, num_data_points=100, image_size=10):
        super().__init__(config_creator_fn)
        if not config_creator_fn:
            self._config = self.create_hawq_test_config(batch_size, num_data_points, image_size)
        self.num_data_points = num_data_points
        self.compression_ratio = 0
        self.should_add_flops = False

    def _set_bitwidth_assignment_mode(self, mode: BitwidthAssignmentMode):
        self._config["compression"]["initializer"]["precision"]["bitwidth_assignment_mode"] = mode.value
        self._options["mode"] = str(mode.value)
        return self

    def strict_mode(self):
        return self._set_bitwidth_assignment_mode(BitwidthAssignmentMode.STRICT)

    def liberal_mode(self):
        return self._set_bitwidth_assignment_mode(BitwidthAssignmentMode.LIBERAL)

    def build(self):
        return self._config

    def for_npu(self):
        super().for_npu()
        return self.strict_mode()

    def check_compression_ratio(self, compression_ratio=1.5):
        self.compression_ratio = compression_ratio
        return self

    def add_flops(self):
        self.should_add_flops = True
        return self

    @staticmethod
    def create_hawq_test_config(batch_size=10, num_data_points=100, image_size=10):
        config = get_quantization_config_without_range_init()
        config["input_info"] = {
            "sample_size": [batch_size, 3, image_size, image_size],
        }
        config["batch_size"] = batch_size
        config["compression"].update(
            {
                "initializer": {
                    "precision": {
                        "type": "hawq",
                        "bits": [4, 8, 6],
                        "num_data_points": num_data_points,
                        "iter_number": 1,
                        "tolerance": 1e-2,
                    },
                    "range": {"num_init_samples": 1},
                    "batchnorm_adaptation": {"num_bn_adaptation_samples": 0},
                }
            }
        )
        return config


def ssd_vgg_512_test():
    ssd_params = SampleConfig(
        {
            "steps": [8, 16, 32, 64, 128, 256, 512],
            "min_sizes": [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],
            "max_sizes": [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
            "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
            "variance": [0.1, 0.1, 0.2, 0.2],
            "clip": False,
            "flip": True,
        }
    )
    return SSD_VGG(cfg=ssd_params, size=512, num_classes=21)


def get_avg_traces(model, init_device: str):
    num_layers = len(get_all_modules_by_type(model, ["Conv2d", "Linear"]))
    return torch.randperm(num_layers).to(init_device) + 1


def check_bitwidth_graph(algo_ctrl, model, path_to_dot, graph_dir, add_flops=False):
    if torch.cuda.is_available():
        model = model.cuda()
    all_quantizers_per_full_scope = HAWQDebugger.get_all_quantizers_per_full_scope(model)
    quantizer_switcher = QuantizersSwitcher(list(all_quantizers_per_full_scope.values()))
    # graph may not contain some quantizers (e.g. in staged scenario)
    quantizer_switcher.enable_quantizers()
    model.nncf.rebuild_graph()
    groups_of_adjacent_quantizers = algo_ctrl.groups_of_adjacent_quantizers
    graph = BitwidthGraph(algo_ctrl, model, groups_of_adjacent_quantizers, add_flops).get()
    nx_graph = add_adjust_padding_nodes(graph, model)
    path_to_dot = get_full_path_to_the_graph(path_to_dot, graph_dir)
    compare_nx_graph_with_reference(nx_graph, path_to_dot)


class HAWQTestStruct(NamedTuple):
    model_creator: Callable[[], nn.Module] = mobilenet_v2
    config_builder: HAWQConfigBuilder = HAWQConfigBuilder().for_npu()
    filename_suffix: str = "hw_config_npu"
    avg_traces_creator: Callable[[nn.Module, str], torch.Tensor] = get_avg_traces

    def __str__(self):
        return "_".join([self.model_creator.__name__, str(self.config_builder)])


INCV3_FLOPS_PER_MODULE = [83886080, 100663296, 117440512, 56623104, 56623104, 198180864, 50331648, 56623104, 56623104]

# WARNING: BITWIDTH_PER_MODULE should be set as max(weight_bits, act_bits) since this is how compression
# ratio is calculated inside HAWQ

# Currently the HAWQ sets up  4-bit weights but 8-bit activations for 117440512 module,
# so effective flops would be computed as if the module had 8-bit weights, therefor "[8, 8, 8" instead of "[8, 8, 4"
INCV3_BITWIDTH_PER_MODULE = [8, 8, 8, 8, 4, 4, 4, 4, 8]
INCV3_BITS_COMPLEXITY = map(lambda x, y: x * y, INCV3_FLOPS_PER_MODULE, INCV3_BITWIDTH_PER_MODULE)
INCV3_COMPRESSION_RATIO = sum(INCV3_FLOPS_PER_MODULE) * 8 / sum(INCV3_BITS_COMPLEXITY)

HAWQ_TEST_PARAMS = (
    HAWQTestStruct(config_builder=HAWQConfigBuilder().staged()),
    HAWQTestStruct(config_builder=HAWQConfigBuilder().for_trial()),
    HAWQTestStruct(config_builder=HAWQConfigBuilder().for_cpu()),
    HAWQTestStruct(config_builder=HAWQConfigBuilder().for_npu().liberal_mode().with_ratio(1.5)),
    HAWQTestStruct(config_builder=HAWQConfigBuilder().with_ratio(1.02).for_npu()),
    HAWQTestStruct(
        model_creator=squeezenet1_1, config_builder=HAWQConfigBuilder().with_sample_size([1, 3, 224, 224]).for_npu()
    ),
    HAWQTestStruct(model_creator=resnet50, config_builder=HAWQConfigBuilder().with_ratio(1.11).for_npu()),
    HAWQTestStruct(model_creator=resnet50, config_builder=HAWQConfigBuilder().for_npu().liberal_mode().with_ratio(1.5)),
    HAWQTestStruct(
        model_creator=inception_v3,
        avg_traces_creator=lambda x, y: get_avg_traces(x, y)[:95],
        config_builder=HAWQConfigBuilder().with_sample_size([2, 3, 299, 299]).for_npu().with_ratio(1),
    ),
    HAWQTestStruct(
        model_creator=inception_v3,
        avg_traces_creator=lambda x, y: get_avg_traces(x, y)[:94],
        config_builder=HAWQConfigBuilder()
        .with_sample_size([2, 3, 299, 299])
        .for_npu()
        .liberal_mode()
        .with_ignored_scope(
            ["Inception3/BasicConv2d[Conv2d_2a_3x3]/NNCFConv2d[conv]/conv2d_0"], target_group=QuantizerGroup.WEIGHTS
        )
        .with_ratio(1.5),
    ),
    HAWQTestStruct(
        model_creator=inception_v3,
        avg_traces_creator=lambda x, y: get_avg_traces(x, y)[:9],
        config_builder=HAWQConfigBuilder()
        .with_sample_size([2, 3, 299, 299])
        .for_npu()
        .liberal_mode()
        .with_target_scope([r"{re}.*InceptionE\[Mixed_7c\].*"])
        .with_ratio(1.3)
        .check_compression_ratio(INCV3_COMPRESSION_RATIO)
        .add_flops(),
    ),
    HAWQTestStruct(
        model_creator=inception_v3,
        avg_traces_creator=lambda x, y: get_avg_traces(x, y)[:95],
        config_builder=HAWQConfigBuilder().with_sample_size([2, 3, 299, 299]).for_npu().liberal_mode().with_ratio(1.5),
    ),
    HAWQTestStruct(
        model_creator=ssd_vgg_512_test,
        config_builder=HAWQConfigBuilder().with_sample_size([1, 3, 512, 512]).for_npu().with_ratio(1.09),
    ),
    HAWQTestStruct(
        model_creator=ssd_vgg_512_test,
        config_builder=HAWQConfigBuilder().with_sample_size([1, 3, 512, 512]).for_npu().liberal_mode().with_ratio(1.5),
    ),
)


@pytest.mark.parametrize("params", HAWQ_TEST_PARAMS, ids=[str(p) for p in HAWQ_TEST_PARAMS])
def test_hawq_precision_init(_seed, dataset_dir, tmp_path, mocker, params):
    config_builder = params.config_builder
    config = config_builder.build()

    model = params.model_creator()
    if torch.cuda.is_available():
        model = model.cuda()
        pregen_device = "cuda"
    else:
        pregen_device = "cpu"

    pregen_traces_for_all_layers = params.avg_traces_creator(model, pregen_device)
    criterion = nn.CrossEntropyLoss().cuda()
    if not dataset_dir:
        dataset_dir = str(tmp_path)
    train_loader, _ = create_test_dataloaders(config, dataset_dir)
    config = register_default_init_args(config, train_loader, criterion=criterion)

    mocked_trace = mocker.patch(
        "nncf.torch.quantization.hessian_trace.HessianTraceEstimator.get_average_traces", autospec=True
    )
    ratio_list_spy = mocker.spy(HAWQPrecisionInitializer, "get_compression_ratio_per_qconfig_sequence")
    chosen_index_spy = mocker.spy(HAWQPrecisionInitializer, "choose_qconfig_sequence")

    # There may be less traces required to be calculated during HAWQ than there are weightable layers.
    def side_effect_fn(self, max_iter=500, tolerance=1e-5):
        return pregen_traces_for_all_layers[: len(self._parameter_handler.parameters)]

    mocked_trace.side_effect = side_effect_fn
    model, ctrl = create_compressed_model_and_algo_for_test(model, config)

    path_to_dot = "{}_{}.dot".format(params.model_creator.__name__, config_builder.filename_suffix())
    graph_dir = os.path.join("quantized", "hawq")
    check_bitwidth_graph(ctrl, model, path_to_dot, graph_dir, add_flops=config_builder.should_add_flops)
    if config_builder.compression_ratio:
        ratio_list = ratio_list_spy.spy_return
        index = chosen_index_spy.spy_return
        assert config_builder.compression_ratio == ratio_list[index]


class RefRatios(NamedTuple):
    target_ratio: int
    expected_ratio: int

    def __str__(self):
        return f"target_ratio:{str(self.target_ratio)}__expected_ratio:{str(self.expected_ratio)}"


TEST_REF_RATIOS = [RefRatios(1, 2), RefRatios(2, 2), RefRatios(3, 4), RefRatios(4, 4), RefRatios(5, 6), RefRatios(6, 6)]


@pytest.mark.parametrize("ratios", TEST_REF_RATIOS, ids=map(str, TEST_REF_RATIOS))
def test_can_choose_pareto_optimal_sequence(ratios):
    # (metric)
    # 6|   *
    # 5| *
    # 4|           *
    # 3|     *
    # 2|       *
    # 1|   *
    #    _ _ _ _ _ _
    #    1 2 3 4 5 6  (ratio)
    compression_ratio_per_qconfig = [1, 2, 2, 3, 4, 6]
    metric_per_qconfig_sequences = [5, 1, 6, 3, 2, 4]
    target_ratio, expected_ratio = ratios
    metric_per_qconfig_sequences = list(map(lambda x: torch.Tensor([x]), metric_per_qconfig_sequences))

    qconfig_sequence_index = HAWQPrecisionInitializer.choose_qconfig_sequence(
        metric_per_qconfig_sequences, compression_ratio_per_qconfig, target_ratio
    )

    assert compression_ratio_per_qconfig[qconfig_sequence_index] == expected_ratio


def test_hawq_hw_npu_config_e2e(_seed, dataset_dir, tmp_path):
    config = HAWQConfigBuilder().for_npu().liberal_mode().with_ratio(1.5).build()
    model = MobileNetV2(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    if not dataset_dir:
        dataset_dir = str(tmp_path)
    train_loader, _ = create_test_dataloaders(config, dataset_dir)
    config = register_default_init_args(config, train_loader, criterion=criterion)

    create_compressed_model_and_algo_for_test(model, config)


HAWQTestParams = namedtuple(
    "HAWQTestParams", ("iter_number", "batch_size", "num_data_points", "cuda_ref_trace", "cpu_ref_trace")
)


@pytest.mark.parametrize(
    "params",
    (
        HAWQTestParams(200, 13, 100, 1.2741253547860323, 1.274125503581261),
        HAWQTestParams(2, 13, 100, 1.2646427814393832, 1.2646428162034615),
        HAWQTestParams(2, 10, 10, 1.830527384351921, 1.8305243724338203),
        HAWQTestParams(2, 10, 5, 1.830527384351921, 1.8305243724338203),
    ),
    ids=("until_threshold", "until_num_iter", "batch_eq_num_data", "batch_larger_num_data"),
)
def test_hawq_on_single_conv_without_quantizers(_seed, dataset_dir, tmp_path, params: HAWQTestParams, mocker):
    config = get_squeezenet_quantization_config(batch_size=params.batch_size)
    iter_number = params.iter_number
    tolerance = 4e-4

    model = squeezenet1_1(num_classes=10, dropout=0)

    from torchvision.models import SqueezeNet1_1_Weights

    load_state(model, SqueezeNet1_1_Weights.IMAGENET1K_V1.get_state_dict(progress=False))
    criterion = nn.CrossEntropyLoss()
    ref_trace = params.cpu_ref_trace
    rtol = 1e-5
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        ref_trace = params.cuda_ref_trace
        rtol = 1e-6

    if not dataset_dir:
        dataset_dir = str(tmp_path)
    data_loader, _ = create_test_dataloaders(config, dataset_dir)
    device = get_model_device(model)

    for _, param in model.named_parameters():
        param.requires_grad = False
    first_conv = next(iter(get_all_modules_by_type(model, "Conv2d").values()))
    first_conv.weight.requires_grad = True
    ph_import = "nncf.torch.quantization.hessian_trace.ParameterHandler"
    sample_rademacher_patch = mocker.patch(f"{ph_import}.sample_rademacher_like_params", autospec=True)
    sample_normal_patch = mocker.patch(f"{ph_import}.sample_normal_like_params", autospec=True)

    def mock_sampling_fn(self):
        return list(map(lambda x: torch.from_numpy(random_sample(x.shape)).to(device=self._device), self.parameters))

    sample_rademacher_patch.side_effect = mock_sampling_fn
    sample_normal_patch.side_effect = mock_sampling_fn

    trace_estimator = HessianTraceEstimator(
        model, default_criterion_fn, criterion, device, data_loader, params.num_data_points
    )
    actual_state = trace_estimator.get_average_traces(max_iter=iter_number, tolerance=tolerance)
    assert math.isclose(actual_state.item(), ref_trace, rel_tol=rtol)


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


def get_skipped_quantized_weight_node_names() -> List[NNCFNodeName]:
    scopes_list = [
        "MobileNetV2/Sequential[features]/Conv2dNormActivation[18]/NNCFConv2d[0]/conv2d_0",
        "MobileNetV2/Sequential[features]/InvertedResidual[17]/Sequential[conv]/NNCFConv2d[2]/conv2d_0",
        "MobileNetV2/Sequential[features]/InvertedResidual[16]/Sequential[conv]/NNCFConv2d[2]/conv2d_0",
    ]
    return scopes_list


def test_disable_quantizer_gradients():
    _, parameters_to_restore, model, *_ = disable_quantizer_gradients()
    assert len(parameters_to_restore.originally_disabled_gradients) == 353
    assert len(parameters_to_restore.skipped_gradients_to_enable) == 2
    actual_requires_grad_per_param = get_requires_grad_per_param(model)
    path_to_ref = str(TEST_ROOT / "torch/data/hawq_reference/mobilenet_v2_requires_grad_per_param.json")
    compare_with_ref_if_exists(actual_requires_grad_per_param, path_to_ref)


def test_enable_quantizer_gradients():
    switcher, params_to_restore, model, ctrl, origi_requires_grad_per_param = disable_quantizer_gradients()
    quantized_modules = ctrl.weight_quantizers
    HAWQPrecisionInitializer.restore_disabled_gradients(switcher, model, quantized_modules, params_to_restore)
    actual_requires_grad_per_param = get_requires_grad_per_param(model)
    assert origi_requires_grad_per_param == actual_requires_grad_per_param


def disable_quantizer_gradients():
    config = get_quantization_config_without_range_init()
    config["input_info"] = {
        "sample_size": [2, 3, 10, 10],
    }
    register_bn_adaptation_init_args(config)
    model = MobileNetV2(num_classes=10)
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    original_requires_grad_per_param = get_requires_grad_per_param(model)
    quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
    all_quantizations = get_all_modules_by_type(model, quantization_types)
    quantizers_switcher = QuantizersSwitcher(list(all_quantizations.values()))
    params_to_restore = HAWQPrecisionInitializer.disable_all_gradients_except_weights_of_quantized_modules(
        quantizers_switcher, compression_ctrl.weight_quantizers, model, get_skipped_quantized_weight_node_names()
    )
    return quantizers_switcher, params_to_restore, model, compression_ctrl, original_requires_grad_per_param


def get_path_to_bitwidth_dump(tmp_path, rank):
    out_file_path = tmp_path / "bitwidth_per_scope_gpu{}.pt".format(rank)
    return out_file_path


def precision_init_dumping_worker(gpu, ngpus_per_node, config, tmp_path):
    distributed_init_test_default(gpu, ngpus_per_node, config)
    data_loader = create_rank_dataloader(config, gpu)
    model = safe_thread_call(partial(mobilenet_v2, pretrained=True))
    model.eval()
    criterion = torch.nn.MSELoss().cuda(config.gpu)
    config = register_default_init_args(
        config, data_loader, criterion=criterion, autoq_eval_fn=lambda *x: 0, val_loader=data_loader
    )
    quant_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    quant_model = post_compression_test_distr_init(compression_ctrl, config, ngpus_per_node, quant_model)

    # just to reproduce the same scale values without Dropout
    quant_model.eval()

    act_bitwidth_per_scope = get_bitwidth_per_scope(quant_model.module)
    out_file_path = get_path_to_bitwidth_dump(tmp_path, config.rank)
    torch.save(act_bitwidth_per_scope, str(out_file_path))


@pytest.mark.cuda
def test_can_broadcast_initialized_precisions_in_distributed_mode(tmp_path, runs_subprocess_in_precommit):
    if not torch.cuda.is_available():
        pytest.skip("Skipping CUDA test cases for CPU only setups")
    config_builder = HAWQConfigBuilder(batch_size=2, num_data_points=10).for_trial()
    config = config_builder.build()
    ngpus_per_node = torch.cuda.device_count()
    config.world_size = ngpus_per_node
    torch.multiprocessing.spawn(
        precision_init_dumping_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, tmp_path), join=True
    )

    assert not compare_multi_gpu_dump(config, tmp_path, get_path_to_bitwidth_dump)


@pytest.mark.parametrize(("method_name", "expected_behavior"), [("_calc_traces", pytest.raises(nncf.InternalError))])
def test_hawq_behaviour__if_method_returns_none(mocker, method_name, expected_behavior):
    config = HAWQConfigBuilder().with_sample_size([1, 1, 4, 4]).for_trial().build()
    config["compression"]["initializer"]["range"]["num_init_samples"] = 0
    model = BasicConvTestModel()
    mock_train_loader = mocker.stub()
    mock_train_loader.batch_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.register_extra_structs(
        [
            QuantizationPrecisionInitArgs(
                criterion_fn=mocker.stub(), criterion=mocker.stub(), data_loader=mock_train_loader, device=device
            )
        ]
    )
    mocker.patch("nncf.common.initialization.batchnorm_adaptation.BatchnormAdaptationAlgorithm.run")
    mocked_calc_traces = mocker.patch(
        "nncf.torch.quantization.precision_init.hawq_init.HAWQPrecisionInitializer._calc_traces"
    )
    stub = mocker.stub()
    stub.traces_order = TracesOrder([0])
    mocked_calc_traces.return_value = stub

    mocked_method = mocker.patch(
        "nncf.torch.quantization.precision_init.hawq_init.HAWQPrecisionInitializer." + method_name
    )
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
    hawq_debugger = HAWQDebugger(
        quantizer_configurations,
        perturbations,
        [weight_observers, weight_observers],
        traces_per_layer,
        [qconf1.num_bits, qconf2.num_bits],
    )

    hawq_debugger.dump_metric_MB(metric_per_qconfig_sequence)
    hawq_debugger.dump_metric_flops(metric_per_qconfig_sequence, flops_per_config, choosen_config_index)
    hawq_debugger.dump_avg_traces()
    hawq_debugger.dump_density_of_quantization_noise()
    hawq_debugger.dump_perturbations_ratio()
    test_dir = tmp_path / Path("hawq_dumps")
    num_dump_files = len([name for name in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, name))])
    assert num_dump_files == 6


def get_quantization_config_with_ignored_scope():
    config = get_quantization_config_without_range_init()
    config["compression"]["ignored_scopes"] = "ConvLinear/NNCFLinear[fc]"
    return config


class RatioCalculatorTestDesc:
    NAMES_OF_INSERTION_POINTS = [
        "/nncf_model_input_0|OUTPUT",
        "ConvLinear/NNCFConv2d[conv1]/conv2d_0|WEIGHT",
        "ConvLinear/NNCFConv2d[conv1]/conv2d_0|OUTPUT",
        "ConvLinear/NNCFLinear[fc]/linear_0|WEIGHT",
    ]

    def __init__(self, ref_ratio: float = 1):
        self._bitwidth_sequence = [8] * len(self.NAMES_OF_INSERTION_POINTS)
        self._config_factory = get_quantization_config_without_range_init
        self._ignored_scopes = []
        self.ref_ratio = ref_ratio

    def bitwidths(self, bitwidth_sequence=List[int]):
        self._bitwidth_sequence = bitwidth_sequence
        return self

    def ignore_fc(self):
        self._ignored_scopes = ["ConvLinear/NNCFLinear[fc]/linear_0"]
        return self

    def create_config(self):
        config = self._config_factory()
        if self._ignored_scopes:
            config["compression"]["ignored_scopes"] = self._ignored_scopes
        return config

    def apply_to_quantizer_setup(self, quantizer_setup: SingleConfigQuantizerSetup) -> SingleConfigQuantizerSetup:
        for i, bitwidth in enumerate(self._bitwidth_sequence):
            ip_name = self.NAMES_OF_INSERTION_POINTS[i]
            quantization_points = quantizer_setup.quantization_points.values()
            found_qp = list(filter(lambda qp: str(qp.insertion_point) == ip_name, quantization_points))
            assert len(found_qp) == 1
            found_qp[0].qconfig.num_bits = bitwidth
        return quantizer_setup

    def __str__(self):
        is_ignored = "with_FC_ignored" if self._ignored_scopes else "all"
        return "_".join([is_ignored, *map(str, self._bitwidth_sequence)])


class ConvLinear(nn.Module):
    CONV_FLOPS = 72
    LINEAR_FLOPS = 108

    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 1, 2, -1, -2)
        self.fc = nn.Linear(3, 6)

    def forward(self, x):
        return self.fc(self.conv1(x))


CONV_FLOPS = ConvLinear.CONV_FLOPS
LINEAR_FLOPS = ConvLinear.LINEAR_FLOPS
MAX_BITS_COMPLEXITY = (CONV_FLOPS + LINEAR_FLOPS) * 8
R48 = MAX_BITS_COMPLEXITY / (CONV_FLOPS * 4 + LINEAR_FLOPS * 8)
R84 = MAX_BITS_COMPLEXITY / (CONV_FLOPS * 8 + LINEAR_FLOPS * 4)

RATIO_CALCULATOR_TEST_DESCS = [
    RatioCalculatorTestDesc(ref_ratio=2.0).bitwidths([4, 4, 4, 4]),
    RatioCalculatorTestDesc(ref_ratio=R48).bitwidths([4, 4, 4, 8]),
    RatioCalculatorTestDesc(ref_ratio=R48).bitwidths([4, 4, 8, 4]),
    RatioCalculatorTestDesc(ref_ratio=R48).bitwidths([4, 4, 8, 8]),
    RatioCalculatorTestDesc(ref_ratio=R84).bitwidths([4, 8, 4, 4]),
    RatioCalculatorTestDesc(ref_ratio=1.0).bitwidths([4, 8, 4, 8]),
    RatioCalculatorTestDesc(ref_ratio=1.0).bitwidths([4, 8, 8, 4]),
    RatioCalculatorTestDesc(ref_ratio=1.0).bitwidths([4, 8, 8, 8]),
    RatioCalculatorTestDesc(ref_ratio=R84).bitwidths([8, 4, 4, 4]),
    RatioCalculatorTestDesc(ref_ratio=1.0).bitwidths([8, 4, 4, 8]),
    RatioCalculatorTestDesc(ref_ratio=1.0).bitwidths([8, 4, 8, 4]),
    RatioCalculatorTestDesc(ref_ratio=1.0).bitwidths([8, 4, 8, 8]),
    RatioCalculatorTestDesc(ref_ratio=R84).bitwidths([8, 8, 4, 4]),
    RatioCalculatorTestDesc(ref_ratio=1.0).bitwidths([8, 8, 4, 8]),
    RatioCalculatorTestDesc(ref_ratio=1.0).bitwidths([8, 8, 8, 4]),
    RatioCalculatorTestDesc(ref_ratio=1.0).bitwidths([8, 8, 8, 8]),
    RatioCalculatorTestDesc(ref_ratio=2.0).bitwidths([4, 4]).ignore_fc(),
    RatioCalculatorTestDesc(ref_ratio=1.0).bitwidths([4, 8]).ignore_fc(),
    RatioCalculatorTestDesc(ref_ratio=1.0).bitwidths([8, 4]).ignore_fc(),
    RatioCalculatorTestDesc(ref_ratio=1.0).bitwidths([8, 8]).ignore_fc(),
]


@pytest.mark.parametrize("desc", RATIO_CALCULATOR_TEST_DESCS, ids=map(str, RATIO_CALCULATOR_TEST_DESCS))
def test_compression_ratio(desc, mocker):
    config = desc.create_config()
    register_bn_adaptation_init_args(config)
    from nncf.torch.quantization.algo import QuantizationBuilder

    get_single_config_quantizer_setup_spy = mocker.spy(QuantizationBuilder, "_get_single_config_quantizer_setup")
    model, ctrl = create_compressed_model_and_algo_for_test(ConvLinear(), config)

    quantizer_setup = get_single_config_quantizer_setup_spy.spy_return
    weight_qp_id_per_activation_qp_id = ctrl.groups_of_adjacent_quantizers.weight_qp_id_per_activation_qp_id
    flops_per_module = model.nncf.get_flops_per_module()
    ratio_calculator = CompressionRatioCalculator(flops_per_module, quantizer_setup, weight_qp_id_per_activation_qp_id)

    quantizer_setup = desc.apply_to_quantizer_setup(quantizer_setup)
    assert ratio_calculator.run_for_quantizer_setup(quantizer_setup) == desc.ref_ratio


def test_staged_quantization_saves_enabled_quantizers_in_state_dict(tmp_path):
    config = get_quantization_config_without_range_init()
    config["compression"]["params"] = {"activations_quant_start_epoch": 2, "weights_quant_start_epoch": 1}
    register_bn_adaptation_init_args(config)
    _, ctrl_save = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config)
    ctrl_save.scheduler.epoch_step()
    ctrl_save.scheduler.epoch_step()
    compression_state = ctrl_save.get_compression_state()
    _, ctrl_load = create_compressed_model_and_algo_for_test(
        BasicConvTestModel(), config, compression_state=compression_state
    )
    for quantizer_info in ctrl_load.non_weight_quantizers.values():
        assert not quantizer_info.quantizer_module_ref.is_enabled_quantization()
    for quantizer_info in ctrl_load.weight_quantizers.values():
        assert quantizer_info.quantizer_module_ref.is_enabled_quantization()
