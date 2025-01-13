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
import re
from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple, Union

import pytest
import torch
import torch.utils.data
from pytest import approx
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import squeezenet1_1

import nncf
import nncf.torch.tensor_statistics.collectors as pt_collectors
from nncf.common.graph import NNCFNodeName
from nncf.common.quantization.initialization.range import PerLayerRangeInitConfig
from nncf.common.quantization.initialization.range import RangeInitConfig
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.common.quantization.quantizer_setup import WeightQuantizationInsertionPoint
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.config import NNCFConfig
from nncf.config.structures import QuantizationRangeInitArgs
from nncf.tensor import Tensor
from nncf.torch import utils
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.initialization import DefaultInitializingDataLoader
from nncf.torch.initialization import wrap_dataloader_for_init
from nncf.torch.quantization.external_quantizer import EXTERNAL_QUANTIZERS_STORAGE_NAME
from nncf.torch.quantization.init_range import PTRangeInitCollectorParams
from nncf.torch.quantization.init_range import PTRangeInitParams
from nncf.torch.quantization.init_range import StatCollectorGenerator
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.tensor_statistics.statistics import pt_convert_stat_to_min_max_tensor_stat
from nncf.torch.utils import get_all_modules_by_type
from nncf.torch.utils import safe_thread_call
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.helpers import get_empty_config
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.quantization.quantization_helpers import compare_multi_gpu_dump
from tests.torch.quantization.quantization_helpers import create_rank_dataloader
from tests.torch.quantization.quantization_helpers import distributed_init_test_default
from tests.torch.quantization.quantization_helpers import get_squeezenet_quantization_config
from tests.torch.quantization.quantization_helpers import post_compression_test_distr_init


def scale_signed_dumping_worker(gpu, ngpus_per_node, config, tmp_path):
    distributed_init_test_default(gpu, ngpus_per_node, config)
    data_loader = create_rank_dataloader(config, gpu)
    model = safe_thread_call(partial(squeezenet1_1, pretrained=True))

    config.register_extra_structs([QuantizationRangeInitArgs(wrap_dataloader_for_init(data_loader))])
    quant_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    compression_scheduler = compression_ctrl.scheduler

    quant_model = post_compression_test_distr_init(compression_ctrl, config, ngpus_per_node, quant_model)

    criterion = torch.nn.MSELoss().cuda(config.gpu)
    optimizer = torch.optim.Adam(quant_model.parameters(), lr=0.01)

    torch.backends.cudnn.benchmark = True

    # just to reproduce the same scale values without Dropout
    quant_model.eval()

    act_sum = 0
    for layer in get_all_modules_by_type(quant_model, "SymmetricQuantizer").values():
        act_sum += layer.scale.sum()
    ref_sum = 3720.864
    assert act_sum.item() == approx(ref_sum, 0.01), "sum of scales is not expected {} vs {} rank {}".format(
        act_sum.item(), ref_sum, config.rank
    )

    out_file_path = get_path_after_broadcast(tmp_path, config.rank)
    save_params(quant_model, out_file_path)
    compression_scheduler.step()
    for i, (input_, _) in enumerate(data_loader):
        if i > 5:
            break
        output = quant_model(input_)
        optimizer.zero_grad()
        dummy_target = torch.randn(1000).cuda(config.gpu, non_blocking=True)
        loss = criterion(output, dummy_target)
        compression_scheduler.step()
        loss.backward()
        optimizer.step()
        compression_scheduler.step()

    out_file_path = get_path_path_after_train_iters(tmp_path, config.rank)
    save_params(quant_model, out_file_path)


def get_path_path_after_train_iters(tmp_path, rank):
    out_file_path = tmp_path / "scale_signed_after_1_train_iter_gpu{}.pt".format(rank)
    return out_file_path


def get_path_after_broadcast(tmp_path, rank):
    out_file_path = tmp_path / "scale_signed_after_broadcast_gpu{}.pt".format(rank)
    return out_file_path


def save_params(model, out_file_path):
    gpu_scale_signed_params = []
    for _, layer in utils.get_all_modules_by_type(model, "SymmetricQuantizer").items():
        gpu_scale_signed_params.append(
            (layer.scale.to(torch.device("cpu")), layer.signed_tensor.to(torch.device("cpu")))
        )
    with out_file_path.open("wb") as out_file:
        torch.save(gpu_scale_signed_params, out_file)


@pytest.mark.cuda
def test_multiprocessing_distributed_shares_init_scales_signedness_across_gpus(tmp_path, runs_subprocess_in_precommit):
    if not torch.cuda.is_available():
        pytest.skip("Skipping CUDA test cases for CPU only setups")
    num_init_samples = 10

    config = get_squeezenet_quantization_config()
    config["compression"]["initializer"] = {"range": {"num_init_samples": num_init_samples}}

    ngpus_per_node = torch.cuda.device_count()
    config.world_size = ngpus_per_node
    register_bn_adaptation_init_args(config)
    torch.multiprocessing.spawn(
        scale_signed_dumping_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, tmp_path), join=True
    )

    assert not compare_multi_gpu_dump(config, tmp_path, get_path_after_broadcast)
    assert not compare_multi_gpu_dump(config, tmp_path, get_path_path_after_train_iters)


def create_empty_config_without_init_section():
    config = get_empty_config()
    config["compression"] = {"algorithm": "quantization"}
    register_bn_adaptation_init_args(config)
    return config


def create_config():
    config = get_empty_config()
    config["compression"] = {"algorithm": "quantization", "initializer": {"range": {"num_init_samples": 1}}}
    register_bn_adaptation_init_args(config)
    return config


def generate_qp(
    node_name: NNCFNodeName, target: QuantizerGroup, input_port_id: int = None
) -> SingleConfigQuantizationPoint:
    if target is QuantizerGroup.WEIGHTS:
        qip = WeightQuantizationInsertionPoint(target_node_name=node_name)
    elif target is QuantizerGroup.ACTIVATIONS:
        qip = ActivationQuantizationInsertionPoint(target_node_name=node_name, input_port_id=input_port_id)
    else:
        raise nncf.InvalidQuantizerGroupError(
            f"Invalid quantizer group: {target}. "
            f"Supported groups are {QuantizerGroup.WEIGHTS}"
            f"and {QuantizerGroup.ACTIVATIONS}."
        )
    return SingleConfigQuantizationPoint(qip, QuantizerConfig(), [node_name])


@pytest.mark.parametrize("wrap_dataloader", [True], ids=["wrapped_dataloader"])
class TestRangeInit:
    @staticmethod
    def create_algo_and_compressed_model(config):
        model = TwoConvTestModel()
        compressed_model, algo = create_compressed_model_and_algo_for_test(model, config)
        return algo, compressed_model

    @staticmethod
    def create_dataloader(wrap_dataloader, config, num_samples=1) -> DataLoader:
        data_loader = create_ones_mock_dataloader(config, num_samples)
        if wrap_dataloader:
            data_loader = DefaultInitializingDataLoader(data_loader)
        return data_loader

    @staticmethod
    def check_sign_and_scale(model, ref_table):
        model_conv = get_all_modules_by_type(model, "SymmetricQuantizer")
        for scope, module in model_conv.items():
            for pattern, ref_values in ref_table.items():
                match = re.search(pattern, str(scope))
                if match:
                    assert isinstance(module, SymmetricQuantizer)
                    assert module.signed == ref_values[0], "sign is not matched for {}".format(str(scope))
                    assert all(module.scale == ref_values[1]), "scale is not matched for {}".format(str(scope))

    @pytest.mark.parametrize("config_creator", (create_config, create_empty_config_without_init_section))
    def test_scale_and_sign_init_for_quant_algo__without_init_section(self, wrap_dataloader, config_creator):
        config = config_creator()
        data_loader = self.create_dataloader(wrap_dataloader, config)
        config.register_extra_structs([QuantizationRangeInitArgs(data_loader)])
        _, compressed_model = self.create_algo_and_compressed_model(config)

        self.check_sign_and_scale(
            compressed_model,
            {
                ".*Sequential\\[0\\].*UpdateWeight.*": (True, torch.ones(2, 1, 1, 1)),
                ".*Sequential\\[1\\].*UpdateWeight. *": (True, 1),
                ".*activation_quantizers.*Sequential\\[0\\].*": (True, 4),
                ".*activation_quantizers.*nncf_model_input*": (False, 1),
            },
        )

    def test_scale_and_sign_init_for_quant_algo__with_zero_init_steps(self, wrap_dataloader):
        config = create_config()
        config["compression"]["initializer"]["range"]["num_init_samples"] = 0

        data_loader = self.create_dataloader(wrap_dataloader, config)
        config.register_extra_structs([QuantizationRangeInitArgs(data_loader)])
        _, compressed_model = self.create_algo_and_compressed_model(config)

        self.check_sign_and_scale(
            compressed_model,
            {
                ".*Sequential\\[0\\].*UpdateWeight.*": (True, torch.ones(2, 1, 1, 1)),
                ".*Sequential\\[1\\].*UpdateWeight. *": (True, 1),
                ".*activation_quantizers.*Sequential\\[0\\].*": (False, 1),
                ".*activation_quantizers.*nncf_model_input*": (False, 1),
            },
        )

    def test_scale_and_sign_init_for_quant_algo__after_load_state(self, wrap_dataloader):
        config = create_config()
        data_loader = self.create_dataloader(wrap_dataloader, config)
        config.register_extra_structs([QuantizationRangeInitArgs(data_loader)])
        _, compressed_model = self.create_algo_and_compressed_model(config)
        ref_loaded_scale_val = torch.ones((1, 1, 1, 1)) * 100
        load_state(
            compressed_model,
            {
                "module.features.0.0.pre_ops.0.op.signed_tensor": torch.tensor(
                    [0.0]
                ),  # quantizer of 1st conv's weights
                "module.features.1.0.pre_ops.0.op.scale": ref_loaded_scale_val,  # quantizer of 2nd conv's weights
            },
        )

        self.check_sign_and_scale(
            compressed_model,
            {
                ".*Sequential\\[0\\].*UpdateWeight.*": (False, torch.ones(2, 1, 1, 1)),
                ".*Sequential\\[1\\].*UpdateWeight. *": (True, ref_loaded_scale_val),
                ".*activation_quantizers.*Sequential\\[0\\].*": (True, 4),
                ".*activation_quantizers.*nncf_model_input*": (False, 1),
            },
        )

    def test_scope_overrides(self, wrap_dataloader):
        config = create_config()
        config["target_device"] = "TRIAL"
        config["compression"]["scope_overrides"] = {
            "weights": {
                r"{re}NNCFConv2d\[[0-9]*\]/conv2d_0": {
                    "bits": 7,
                    "mode": "asymmetric",
                },
            },
            "activations": {
                r"{re}NNCFConv2d\[[0-9]*\]/conv2d_0": {
                    "bits": 7,
                    "signed": False,
                }
            },
        }
        data_loader = self.create_dataloader(wrap_dataloader, config)
        config.register_extra_structs([QuantizationRangeInitArgs(data_loader)])
        _, compressed_model = self.create_algo_and_compressed_model(config)

        quantizers = get_all_modules_by_type(compressed_model, ["SymmetricQuantizer", "AsymmetricQuantizer"])
        quantizer_str_dict = {str(k): v for k, v in quantizers.items()}
        group_1 = [
            quantizer_str_dict[
                "TwoConvTestModel/Sequential[features]/"
                "Sequential[0]/NNCFConv2d[0]/ModuleDict[pre_ops]/UpdateWeight[0]/"
                "AsymmetricQuantizer[op]"
            ],
            quantizer_str_dict[
                "TwoConvTestModel/Sequential[features]/"
                "Sequential[1]/NNCFConv2d[0]/ModuleDict[pre_ops]/UpdateWeight[0]/"
                "AsymmetricQuantizer[op]"
            ],
        ]
        group_2 = [
            quantizer_str_dict[
                f"TwoConvTestModel/NNCFNetworkInterface[_nncf]/"
                f"ModuleDict[{EXTERNAL_QUANTIZERS_STORAGE_NAME}]/"
                "SymmetricQuantizer[TwoConvTestModel/Sequential[features]"
                "/Sequential[0]/NNCFConv2d[0]/conv2d_0|OUTPUT]"
            ],
            quantizer_str_dict[
                f"TwoConvTestModel/NNCFNetworkInterface[_nncf]/"
                f"ModuleDict[{EXTERNAL_QUANTIZERS_STORAGE_NAME}]/SymmetricQuantizer"
                "[/nncf_model_input_0|OUTPUT]"
            ],
        ]

        for quantizer in group_1:
            assert isinstance(quantizer, AsymmetricQuantizer)
            assert quantizer.levels == 2**7
        for quantizer in group_2:
            assert isinstance(quantizer, SymmetricQuantizer)
            assert not quantizer.signed

    PerLayerRangeInitTestStruct = namedtuple(
        "PerLayerRangeInitTestStruct", ("range_init_config", "qps_vs_expected_init_config")
    )

    PER_LAYER_RANGE_INIT_TEST_CASES = [
        PerLayerRangeInitTestStruct(
            range_init_config=[{"type": "min_max", "num_init_samples": 1, "target_scopes": ["{re}.*"]}],
            qps_vs_expected_init_config=[
                (
                    generate_qp(
                        "/nncf_model_input_0",
                        QuantizerGroup.ACTIVATIONS,
                    ),
                    RangeInitConfig(init_type="min_max", num_init_samples=1),
                ),
                (
                    generate_qp(
                        "TwoConvTestModel/Sequential[features]/Sequential[0]/NNCFConv2d[0]/conv2d_0",
                        QuantizerGroup.ACTIVATIONS,
                    ),
                    RangeInitConfig(init_type="min_max", num_init_samples=1),
                ),
                (
                    generate_qp(
                        "TwoConvTestModel/Sequential[features]/Sequential[1]/NNCFConv2d[0]/conv2d_0",
                        QuantizerGroup.WEIGHTS,
                    ),
                    RangeInitConfig(init_type="min_max", num_init_samples=1),
                ),
            ],
        ),
        PerLayerRangeInitTestStruct(
            range_init_config=[
                {
                    "type": "min_max",
                    "num_init_samples": 1,
                    "target_scopes": ["{re}TwoConvTestModel/Sequential\\[features\\]/.*"],
                },
                {
                    "type": "mean_min_max",
                    "num_init_samples": 2,
                    "ignored_scopes": ["{re}TwoConvTestModel/Sequential\\[features\\]/.*"],
                },
            ],
            qps_vs_expected_init_config=[
                (
                    generate_qp("/nncf_model_input_0", QuantizerGroup.ACTIVATIONS),
                    RangeInitConfig(init_type="mean_min_max", num_init_samples=2),
                ),
                (
                    generate_qp(
                        "TwoConvTestModel/Sequential[features]/Sequential[0]/NNCFConv2d[0]/conv2d_0",
                        QuantizerGroup.ACTIVATIONS,
                    ),
                    RangeInitConfig(init_type="min_max", num_init_samples=1),
                ),
                (
                    generate_qp(
                        "TwoConvTestModel/Sequential[features]/Sequential[0]/NNCFConv2d[0]/conv2d_0",
                        QuantizerGroup.WEIGHTS,
                    ),
                    RangeInitConfig(init_type="min_max", num_init_samples=1),
                ),
                (
                    generate_qp(
                        "TwoConvTestModel/Sequential[features]/Sequential[1]/NNCFConv2d[0]/conv2d_0",
                        QuantizerGroup.ACTIVATIONS,
                    ),
                    RangeInitConfig(init_type="min_max", num_init_samples=1),
                ),
            ],
        ),
        PerLayerRangeInitTestStruct(
            range_init_config=[
                {
                    "type": "min_max",
                    "num_init_samples": 1,
                    "target_quantizer_group": "weights",
                    "target_scopes": ["{re}TwoConvTestModel/Sequential\\[features\\]/.*"],
                },
                {
                    "type": "mean_min_max",
                    "num_init_samples": 2,
                    "ignored_scopes": ["{re}TwoConvTestModel/Sequential\\[features\\]/.*", "{re}/nncf_model_input_0"],
                },
                {
                    "type": "threesigma",
                    "num_init_samples": 1,
                    "target_quantizer_group": "activations",
                    "target_scopes": ["{re}/nncf_model_input_0"],
                },
                {
                    "type": "percentile",
                    "num_init_samples": 10,
                    "params": {"min_percentile": "0.1", "max_percentile": "99.9"},
                    "target_quantizer_group": "activations",
                    "target_scopes": [
                        "TwoConvTestModel/Sequential[features]/Sequential[1]/NNCFConv2d[0]/conv2d_0|OUTPUT"
                    ],
                },
            ],
            qps_vs_expected_init_config=[
                (
                    generate_qp("/nncf_model_input_0", QuantizerGroup.ACTIVATIONS),
                    RangeInitConfig(init_type="threesigma", num_init_samples=1),
                ),
                (
                    generate_qp(
                        "TwoConvTestModel/Sequential[features]/Sequential[0]/NNCFConv2d[0]/conv2d_0",
                        QuantizerGroup.WEIGHTS,
                    ),
                    RangeInitConfig(init_type="min_max", num_init_samples=1),
                ),
                (
                    generate_qp(
                        "TwoConvTestModel/Sequential[features]/Sequential[1]/NNCFConv2d[0]/conv2d_0",
                        QuantizerGroup.ACTIVATIONS,
                    ),
                    RangeInitConfig(
                        init_type="percentile",
                        num_init_samples=10,
                        init_type_specific_params={"min_percentile": "0.1", "max_percentile": "99.9"},
                    ),
                ),
            ],
        ),
    ]

    @staticmethod
    @pytest.fixture(params=PER_LAYER_RANGE_INIT_TEST_CASES)
    def per_layer_range_init_test_struct(request):
        return request.param

    def test_get_init_config_for_quantization_point(self, wrap_dataloader, per_layer_range_init_test_struct):
        per_layer_configs = []
        for sub_init_range_config_dict in per_layer_range_init_test_struct.range_init_config:
            per_layer_configs.append(PerLayerRangeInitConfig.from_dict(sub_init_range_config_dict))

        params = PTRangeInitParams(
            wrap_dataloader, "", global_init_config=None, per_layer_range_init_configs=per_layer_configs
        )

        for qp, ref_range_init_config in per_layer_range_init_test_struct.qps_vs_expected_init_config:
            assert params.get_init_config_for_quantization_point(qp) == ref_range_init_config

    @pytest.mark.parametrize("quant_type", ("symmetric", "asymmetric"))
    def test_ad_hoc_range_init_does_not_replace_parameter_tensors(self, wrap_dataloader, quant_type):
        config = create_config()
        config["compression"].update({"activations": {"mode": quant_type}, "weights": {"mode": quant_type}})

        data_loader = self.create_dataloader(wrap_dataloader, config)
        config.register_extra_structs([QuantizationRangeInitArgs(data_loader)])

        model = TwoConvTestModel()
        quant_model, quant_ctrl = create_compressed_model_and_algo_for_test(model, config)
        param_name_vs_id = {name: id(tnsr) for name, tnsr in quant_model.named_parameters()}

        quant_ctrl.init_range()

        for name, param in quant_model.named_parameters():
            assert param_name_vs_id[name] == id(param)


class SingleConv2dIdentityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 3, 1)
        self.conv2d.weight = torch.nn.Parameter(torch.ones_like(self.conv2d.weight))

    def forward(self, input_):
        return self.conv2d(input_)


def _get_init_tensor_for_range_init_test() -> torch.Tensor:
    test_input_sample = torch.ones([3, 100, 100])
    test_input_sample[0] = torch.range(1, 10_000).view((100, 100))
    test_input_sample[1] = test_input_sample[0] * -2
    test_input_sample[2] = test_input_sample[0] * 3
    return test_input_sample


class SingleConv2dSyntheticWeightModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 3, 100)

        with torch.no_grad():
            value = _get_init_tensor_for_range_init_test()
            for i in range(0, 3):
                self.conv2d.weight[:, i] = value

    def forward(self, input_):
        return self.conv2d(input_)


def init_idfn(val):
    if isinstance(val, tuple):
        return val[0]
    return val


@dataclass
class SymQuantizerScaleRef:
    scale: Tuple[float, ...]


@dataclass
class AsymQuantizerScaleRef:
    input_low: Tuple[float, ...]
    input_range: Tuple[float, ...]


@dataclass
class GranularityQuantizerRefs:
    per_channel: Union[SymQuantizerScaleRef, AsymQuantizerScaleRef]
    per_tensor: Union[SymQuantizerScaleRef, AsymQuantizerScaleRef]


@dataclass
class RangeInitTestCase:
    collector_name: str
    weights_refs_symmetric: GranularityQuantizerRefs
    weights_refs_assymetric: GranularityQuantizerRefs
    activations_refs_symmetric: GranularityQuantizerRefs
    activations_refs_assymetric: GranularityQuantizerRefs


@pytest.mark.parametrize(
    "range_init_test_case",
    (
        [
            RangeInitTestCase(
                collector_name="min_max",
                weights_refs_symmetric=GranularityQuantizerRefs(
                    per_channel=SymQuantizerScaleRef(
                        scale=torch.tensor((10000.0, 20000.0, 30000.0)).view(((3, 1, 1, 1)))
                    ),
                    per_tensor=SymQuantizerScaleRef(scale=30000.0),
                ),
                weights_refs_assymetric=GranularityQuantizerRefs(
                    per_channel=AsymQuantizerScaleRef(
                        input_low=torch.tensor((1.0, -20000.0, 3.0)).view(((3, 1, 1, 1))),
                        input_range=torch.tensor((9999.0, 19998.0, 29997.0)).view(((3, 1, 1, 1))),
                    ),
                    per_tensor=AsymQuantizerScaleRef(input_low=-20000.0, input_range=50000.0),
                ),
                activations_refs_symmetric=GranularityQuantizerRefs(
                    per_channel=SymQuantizerScaleRef(
                        scale=torch.tensor((20000.0, 40000.0, 60000.0)).view(((1, 3, 1, 1)))
                    ),
                    per_tensor=SymQuantizerScaleRef(scale=60000.0),
                ),
                activations_refs_assymetric=GranularityQuantizerRefs(
                    per_channel=AsymQuantizerScaleRef(
                        input_low=torch.tensor((1.0, -40000.0, 3.0)).view(((1, 3, 1, 1))),
                        input_range=torch.tensor((19999.0, 39998.0, 59997.0)).view(((1, 3, 1, 1))),
                    ),
                    per_tensor=AsymQuantizerScaleRef(input_low=-40000.0, input_range=100000.0),
                ),
            ),
            RangeInitTestCase(
                collector_name="mixed_min_max",
                weights_refs_symmetric=GranularityQuantizerRefs(
                    per_channel=SymQuantizerScaleRef(
                        scale=torch.tensor((10000.0, 20000.0, 30000.0)).view(((3, 1, 1, 1)))
                    ),
                    per_tensor=SymQuantizerScaleRef(scale=30000.0),
                ),
                weights_refs_assymetric=GranularityQuantizerRefs(
                    per_channel=AsymQuantizerScaleRef(
                        input_low=torch.tensor((1.0, -20000.0, 3.0)).view(((3, 1, 1, 1))),
                        input_range=torch.tensor((9999.0, 19998.0, 29997.0)).view(((3, 1, 1, 1))),
                    ),
                    per_tensor=AsymQuantizerScaleRef(input_low=-20000.0, input_range=50000.0),
                ),
                activations_refs_symmetric=GranularityQuantizerRefs(
                    per_channel=SymQuantizerScaleRef(
                        scale=torch.tensor((20000.0, 40000.0, 60000.0)).view(((1, 3, 1, 1)))
                    ),
                    per_tensor=SymQuantizerScaleRef(scale=45000.0),
                ),
                activations_refs_assymetric=GranularityQuantizerRefs(
                    per_channel=AsymQuantizerScaleRef(
                        input_low=torch.tensor((1.0, -40000.0, 3.0)).view(((1, 3, 1, 1))),
                        input_range=torch.tensor((19999.0, 39998.0, 59997.0)).view(((1, 3, 1, 1))),
                    ),
                    per_tensor=AsymQuantizerScaleRef(input_low=-30000.0, input_range=75000.0),
                ),
            ),
            RangeInitTestCase(
                collector_name="mean_min_max",
                weights_refs_symmetric=GranularityQuantizerRefs(
                    per_channel=SymQuantizerScaleRef(
                        scale=torch.tensor((10000.0, 20000.0, 30000.0)).view(((3, 1, 1, 1)))
                    ),
                    per_tensor=SymQuantizerScaleRef(scale=30000.0),
                ),
                weights_refs_assymetric=GranularityQuantizerRefs(
                    per_channel=AsymQuantizerScaleRef(
                        input_low=torch.tensor((1.0, -20000.0, 3.0)).view(((3, 1, 1, 1))),
                        input_range=torch.tensor((9999.0, 19998.0, 29997.0)).view(((3, 1, 1, 1))),
                    ),
                    per_tensor=AsymQuantizerScaleRef(input_low=-20000.0, input_range=50000.0),
                ),
                activations_refs_symmetric=GranularityQuantizerRefs(
                    per_channel=SymQuantizerScaleRef(
                        scale=torch.tensor((15000.0, 30000.0, 45000.0)).view(((1, 3, 1, 1)))
                    ),
                    per_tensor=SymQuantizerScaleRef(scale=45000.0),
                ),
                activations_refs_assymetric=GranularityQuantizerRefs(
                    per_channel=AsymQuantizerScaleRef(
                        input_low=torch.tensor((1.5, -30000.0, 4.5)).view(((1, 3, 1, 1))),
                        input_range=torch.tensor((14998.5000, 29997.0000, 44995.5000)).view(((1, 3, 1, 1))),
                    ),
                    per_tensor=AsymQuantizerScaleRef(input_low=-30000.0, input_range=75000.0),
                ),
            ),
            RangeInitTestCase(
                collector_name="threesigma",
                weights_refs_symmetric=GranularityQuantizerRefs(
                    per_channel=SymQuantizerScaleRef(
                        scale=torch.tensor((16120.1719, 32240.3438, 48360.5156)).view(((3, 1, 1, 1)))
                    ),
                    per_tensor=SymQuantizerScaleRef(scale=33780.2891),
                ),
                weights_refs_assymetric=GranularityQuantizerRefs(
                    per_channel=AsymQuantizerScaleRef(
                        input_low=torch.tensor((-6119.1719, -32240.3438, -18357.5156)).view(((3, 1, 1, 1))),
                        input_range=torch.tensor((22239.3438, 44478.6875, 66718.0312)).view(((3, 1, 1, 1))),
                    ),
                    per_tensor=AsymQuantizerScaleRef(input_low=-26279.2871, input_range=60059.5781),
                ),
                activations_refs_symmetric=GranularityQuantizerRefs(
                    per_channel=SymQuantizerScaleRef(
                        scale=torch.tensor((21494.4707, 42988.9414, 64483.4141)).view(((1, 3, 1, 1)))
                    ),
                    per_tensor=SymQuantizerScaleRef(scale=52662.1367),
                ),
                activations_refs_assymetric=GranularityQuantizerRefs(
                    per_channel=AsymQuantizerScaleRef(
                        input_low=torch.tensor((-8159.4707, -42988.9414, -24478.4141)).view(((1, 3, 1, 1))),
                        input_range=torch.tensor((29653.9414, 59307.8828, 88961.8281)).view(((1, 3, 1, 1))),
                    ),
                    per_tensor=AsymQuantizerScaleRef(input_low=-42660.1367, input_range=95322.2734),
                ),
            ),
            RangeInitTestCase(
                collector_name="percentile",
                weights_refs_symmetric=GranularityQuantizerRefs(
                    per_channel=SymQuantizerScaleRef(
                        scale=torch.tensor((6789.3213, 13580.6416, 20367.9629)).view(((3, 1, 1, 1)))
                    ),
                    per_tensor=SymQuantizerScaleRef(scale=7776.0),
                ),
                weights_refs_assymetric=GranularityQuantizerRefs(
                    per_channel=AsymQuantizerScaleRef(
                        input_low=torch.tensor((3210.6790, -13580.6416, 9632.0371)).view(((3, 1, 1, 1))),
                        input_range=torch.tensor((3578.6423, 7157.2837, 10735.9258)).view(((3, 1, 1, 1))),
                    ),
                    per_tensor=AsymQuantizerScaleRef(input_low=-740.6420, input_range=8516.6416),
                ),
                activations_refs_symmetric=GranularityQuantizerRefs(
                    per_channel=SymQuantizerScaleRef(
                        scale=torch.tensor((9052.3213, 18108.0000, 27156.9629)).view(((1, 3, 1, 1)))
                    ),
                    per_tensor=SymQuantizerScaleRef(scale=10734.6426),
                ),
                activations_refs_assymetric=GranularityQuantizerRefs(
                    per_channel=AsymQuantizerScaleRef(
                        input_low=torch.tensor((4280.6792, -18108.0000, 12842.0371)).view(((1, 3, 1, 1))),
                        input_range=torch.tensor((4771.6421, 9544.0000, 14314.9258)).view(((1, 3, 1, 1))),
                    ),
                    per_tensor=AsymQuantizerScaleRef(input_low=-988.0, input_range=11722.6426),
                ),
            ),
            RangeInitTestCase(
                collector_name="mean_percentile",
                weights_refs_symmetric=GranularityQuantizerRefs(
                    per_channel=SymQuantizerScaleRef(
                        scale=torch.tensor((9990.0010, 19980.0020, 29970.0039)).view(((3, 1, 1, 1)))
                    ),
                    per_tensor=SymQuantizerScaleRef(scale=29910.0039),
                ),
                weights_refs_assymetric=GranularityQuantizerRefs(
                    per_channel=AsymQuantizerScaleRef(
                        input_low=torch.tensor((10.999, -19980.0, 32.997)).view(((3, 1, 1, 1))),
                        input_range=torch.tensor((9979.0020, 19958.0039, 29937.0078)).view(((3, 1, 1, 1))),
                    ),
                    per_tensor=AsymQuantizerScaleRef(input_low=-19940.0020, input_range=49850.0078),
                ),
                activations_refs_symmetric=GranularityQuantizerRefs(
                    per_channel=SymQuantizerScaleRef(
                        scale=torch.tensor((14985.0020, 29970.0039, 44955.0078)).view(((1, 3, 1, 1)))
                    ),
                    per_tensor=SymQuantizerScaleRef(scale=44865.0078),
                ),
                activations_refs_assymetric=GranularityQuantizerRefs(
                    per_channel=AsymQuantizerScaleRef(
                        input_low=torch.tensor((16.498, -2.9970e04, 49.496)).view(((1, 3, 1, 1))),
                        input_range=torch.tensor((14968.5039, 29937.0078, 44905.5117)).view(((1, 3, 1, 1))),
                    ),
                    per_tensor=AsymQuantizerScaleRef(input_low=-29910.0039, input_range=74775.0156),
                ),
            ),
        ]
    ),
    ids=init_idfn,
)
def test_init_ranges_are_set(
    quantization_mode: str,
    is_per_channel: bool,
    range_init_test_case: RangeInitTestCase,
):
    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self):
            super().__init__()
            self._length = 2

        def __getitem__(self, idx):
            if idx >= self._length:
                raise StopIteration
            test_input_sample = _get_init_tensor_for_range_init_test() * (idx + 1)
            return test_input_sample, test_input_sample

        def __len__(self):
            return self._length

    data_loader = torch.utils.data.DataLoader(SyntheticDataset(), batch_size=1, drop_last=True)

    range_init_type = range_init_test_case.collector_name
    config_with_init = NNCFConfig()
    config_with_init.update(
        {
            "input_info": {"sample_size": [1, 3, 100, 100]},
            "target_device": "TRIAL",
            "compression": {
                "algorithm": "quantization",
                "activations": {"mode": quantization_mode, "per_channel": is_per_channel},
                "weights": {"mode": quantization_mode, "per_channel": is_per_channel},
                "initializer": {"range": {"num_init_samples": 2, "type": range_init_type}},
            },
        }
    )

    if range_init_type == "percentile":
        config_with_init["compression"]["initializer"]["range"]["params"] = {
            "min_percentile": 32.10,
            "max_percentile": 67.89,
        }

    # Activations init check
    id_model = SingleConv2dIdentityModel()
    config_with_init.register_extra_structs([QuantizationRangeInitArgs(wrap_dataloader_for_init(data_loader))])
    register_bn_adaptation_init_args(config_with_init)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(id_model, config_with_init)

    act_quantizer_info = next(iter(compression_ctrl.non_weight_quantizers.values()))

    if is_per_channel:
        ref_scale = range_init_test_case.activations_refs_symmetric.per_channel.scale
        ref_input_low = range_init_test_case.activations_refs_assymetric.per_channel.input_low
        ref_input_range = range_init_test_case.activations_refs_assymetric.per_channel.input_range
    else:
        ref_scale = range_init_test_case.activations_refs_symmetric.per_tensor.scale
        ref_input_low = range_init_test_case.activations_refs_assymetric.per_tensor.input_low
        ref_input_range = range_init_test_case.activations_refs_assymetric.per_tensor.input_range

    def check_scales(quantizer: BaseQuantizer, per_channel: bool):
        # Absolute tolerance is 1.0 due to percentile value interpolation
        if quantization_mode == "symmetric":
            assert torch.allclose(quantizer.scale, torch.tensor(ref_scale), atol=1.0)
            if per_channel:
                assert quantizer.scale.numel() == 3
            else:
                assert quantizer.scale.numel() == 1
        else:
            assert torch.allclose(quantizer.input_low, torch.tensor(ref_input_low), atol=1.0)

            assert torch.allclose(
                quantizer.input_range,
                torch.tensor(ref_input_range),
                atol=1.0,
            )
            if per_channel:
                assert quantizer.input_low.numel() == 3
                assert quantizer.input_range.numel() == 3
            else:
                assert quantizer.input_low.numel() == 1
                assert quantizer.input_range.numel() == 1

    check_scales(act_quantizer_info.quantizer_module_ref, is_per_channel)
    # Weight init check
    synth_weight_model = SingleConv2dSyntheticWeightModel()
    config_with_init["compression"]["initializer"]["range"]["num_init_samples"] = 1
    _, compression_ctrl = create_compressed_model_and_algo_for_test(synth_weight_model, config_with_init)

    weight_quantizer_info = next(iter(compression_ctrl.weight_quantizers.values()))
    if is_per_channel:
        ref_scale = range_init_test_case.weights_refs_symmetric.per_channel.scale
        ref_input_low = range_init_test_case.weights_refs_assymetric.per_channel.input_low
        ref_input_range = range_init_test_case.weights_refs_assymetric.per_channel.input_range
    else:
        ref_scale = range_init_test_case.weights_refs_symmetric.per_tensor.scale
        ref_input_low = range_init_test_case.weights_refs_assymetric.per_tensor.input_low
        ref_input_range = range_init_test_case.weights_refs_assymetric.per_tensor.input_range

    check_scales(weight_quantizer_info.quantizer_module_ref, is_per_channel)


RangeInitCallCountTestStruct = namedtuple(
    "RangeInitCallCountTestStruct",
    (
        "range_init_config",
        "expected_call_count_initializer_create",
        "expected_call_count_register_input",
    ),
)
RANGE_INIT_CALL_COUNT_TEST_CASES = [
    RangeInitCallCountTestStruct(
        range_init_config={"type": "min_max", "num_init_samples": 5},
        expected_call_count_initializer_create={"min_max": 4, "mean_min_max": 0, "three_sigma": 0},
        expected_call_count_register_input={
            "min_max": 12,  # 2 activation statistics for 5x inputs, 2 weight statistics for 1 input each
            "mean_min_max": 0,
            "three_sigma": 0,
        },
    ),
    RangeInitCallCountTestStruct(
        range_init_config=[
            {
                "type": "min_max",
                "num_init_samples": 5,
                "target_quantizer_group": "weights",
                "target_scopes": ["{re}TwoConvTestModel/Sequential\\[features\\]/.*"],
            },
            {
                "type": "mean_min_max",
                "num_init_samples": 2,
                "ignored_scopes": ["{re}TwoConvTestModel/Sequential\\[features\\]/.*"],
            },
            {
                "type": "threesigma",
                "num_init_samples": 3,
                "target_quantizer_group": "activations",
                "target_scopes": ["{re}TwoConvTestModel/Sequential\\[features\\]/.*"],
            },
        ],
        expected_call_count_initializer_create={"min_max": 2, "mean_min_max": 1, "three_sigma": 1},
        expected_call_count_register_input={
            "min_max": 2,  # Weights only require single input registration
            "mean_min_max": 2,
            "three_sigma": 3,
        },
    ),
]


@pytest.fixture(params=RANGE_INIT_CALL_COUNT_TEST_CASES)
def range_init_call_count_test_struct(request):
    return request.param


class CustomSpy:
    def __init__(self, fn) -> None:
        self._fn = fn
        self.call_count = 0
        self.return_values_list = []

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        retval = self._fn(*args, **kwargs)
        self.return_values_list.append(retval)
        return retval


def test_per_layer_range_init_collectors_are_called_the_required_number_of_times(
    range_init_call_count_test_struct, mocker
):
    range_minmax_init_create_spy = CustomSpy(pt_collectors.get_min_max_statistic_collector)
    mocker.patch("nncf.torch.quantization.init_range.get_min_max_statistic_collector", new=range_minmax_init_create_spy)
    range_meanminmax_init_create_spy = CustomSpy(pt_collectors.get_mixed_min_max_statistic_collector)
    mocker.patch(
        "nncf.torch.quantization.init_range.get_mixed_min_max_statistic_collector", new=range_meanminmax_init_create_spy
    )
    range_threesigma_init_create_spy = CustomSpy(pt_collectors.get_median_mad_statistic_collector)
    mocker.patch(
        "nncf.torch.quantization.init_range.get_median_mad_statistic_collector", new=range_threesigma_init_create_spy
    )

    config = create_config()
    config["compression"]["initializer"]["range"] = range_init_call_count_test_struct.range_init_config
    data_loader = TestRangeInit.create_dataloader(True, config, 10)
    config.register_extra_structs([QuantizationRangeInitArgs(data_loader)])

    TestRangeInit.create_algo_and_compressed_model(config)

    for stat_type, spy in [
        ("min_max", range_minmax_init_create_spy),
        ("mean_min_max", range_meanminmax_init_create_spy),
        ("three_sigma", range_threesigma_init_create_spy),
    ]:
        assert spy.call_count == range_init_call_count_test_struct.expected_call_count_initializer_create[stat_type]
        collected_samples = 0
        for tensor_collector in spy.return_values_list:
            cur_values = set()
            for aggr in tensor_collector.aggregators.values():
                cur_values.add(aggr._collected_samples)
            assert len(cur_values) == 1
            collected_samples += cur_values.pop()

        assert collected_samples == range_init_call_count_test_struct.expected_call_count_register_input[stat_type]


QUANTIZER_RANGE_INITIALIZERS = [
    "min_max",
    "threesigma",
    "mean_min_max",
    "percentile",
    "mixed_min_max",
    "mean_percentile",
]


class QuantizeRangeInitScaleShapeTestStruct:
    def __init__(self, per_channel: bool, is_weights: bool, input_shape: List[int], ref_scale_shape: Tuple[int, ...]):
        self.per_channel = per_channel
        self.is_weights = is_weights
        self.input_shape = input_shape
        self.ref_scale_shape = ref_scale_shape


QRISSTS = QuantizeRangeInitScaleShapeTestStruct

QUANTIZER_RANGE_INIT_TEST_CASES = [
    QRISSTS(per_channel=False, is_weights=False, input_shape=[41, 42, 43, 44], ref_scale_shape=(1,)),
    QRISSTS(per_channel=True, is_weights=False, input_shape=[41, 42, 43, 44], ref_scale_shape=(1, 42, 1, 1)),
    QRISSTS(per_channel=False, is_weights=True, input_shape=[41, 42, 43, 44], ref_scale_shape=(1,)),
    QRISSTS(per_channel=True, is_weights=True, input_shape=[41, 42, 43, 44], ref_scale_shape=(41, 1, 1, 1)),
]


def quantizer_range_init_scale_shape_idfn(fixture_value):
    test_struct: QRISSTS = fixture_value[0]
    postfix = ""
    if test_struct.is_weights:
        postfix += "-W"
    else:
        postfix += "-A"

    if test_struct.per_channel:
        postfix += "-PC"
    else:
        postfix += "-PT"
    return fixture_value[1] + postfix


@pytest.fixture(
    params=itertools.product(QUANTIZER_RANGE_INIT_TEST_CASES, QUANTIZER_RANGE_INITIALIZERS),
    ids=quantizer_range_init_scale_shape_idfn,
)
def quantizer_range_init_test_struct(request):
    return request.param


def test_quantize_range_init_sets_correct_scale_shapes(quantizer_range_init_test_struct: Tuple[QRISSTS, str]):
    test_struct = quantizer_range_init_test_struct[0]
    initializer_type = quantizer_range_init_test_struct[1]
    for quantization_mode in [QuantizationMode.SYMMETRIC, QuantizationMode.ASYMMETRIC]:
        qconfig = PTQuantizerSpec(
            num_bits=8,
            mode=quantization_mode,
            signedness_to_force=None,
            scale_shape=tuple(test_struct.ref_scale_shape),
            narrow_range=test_struct.is_weights,
            half_range=False,
            logarithm_scale=False,
        )
        q_cls = QUANTIZATION_MODULES.get(quantization_mode)
        quantizer: BaseQuantizer = q_cls(qconfig)
        range_init_config = RangeInitConfig(init_type=initializer_type, num_init_samples=1)

        if test_struct.is_weights:
            channel_idx = 0  # channel dim for weights
        else:
            channel_idx = 1  # channel dim for activations

        collector_params = PTRangeInitCollectorParams(
            test_struct.is_weights,
            quantization_mode,
            test_struct.per_channel,
            tuple(test_struct.input_shape),
            channel_idx,
        )

        collector = StatCollectorGenerator.generate_stat_collector_for_range_init_config(
            range_init_config, tuple(quantizer.scale_shape), collector_params
        )
        collector.register_input_for_all_reducers(Tensor(torch.ones(test_struct.input_shape)))
        stat = collector.get_statistics()
        minmax_values = pt_convert_stat_to_min_max_tensor_stat(stat)
        quantizer.apply_minmax_init(min_values=minmax_values.min_values.data, max_values=minmax_values.max_values.data)

        assert quantizer.scale_shape == test_struct.ref_scale_shape
        if quantization_mode == QuantizationMode.SYMMETRIC:
            assert tuple(quantizer.scale.shape) == test_struct.ref_scale_shape
        elif quantization_mode == QuantizationMode.ASYMMETRIC:
            assert tuple(quantizer.input_low.shape) == test_struct.ref_scale_shape
            assert tuple(quantizer.input_range.shape) == test_struct.ref_scale_shape
        else:
            assert False  # options above should be exhaustive


def test_range_initialization_in_train_mode():
    """
    Check that if a model in train mode is being compressed,
    the range initialization statistic collection still runs in eval mode
    """

    class Model(nn.Module):
        def forward(self, x):
            # This forward produces different number of operations depending on
            # the self.training state. If statistics collection was run in
            # training mode it would fail with StatisticsNotCollectedError,
            # because it wouldn't find some nodes discovered during model graph
            # building, which runs in eval mode.
            if self.training:
                return x
            return x * x * x

    config = get_empty_config()
    config["compression"] = {"algorithm": "quantization", "initializer": {"range": {"num_init_samples": 1}}}
    data_loader = wrap_dataloader_for_init(create_ones_mock_dataloader(config, 1))

    config.register_extra_structs([QuantizationRangeInitArgs(data_loader=data_loader)])

    model = Model()
    model.train()
    _, _ = create_compressed_model_and_algo_for_test(model, config)
