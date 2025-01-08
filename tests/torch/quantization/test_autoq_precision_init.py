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
import os
from random import random
from typing import Callable, NamedTuple

import pytest
import torch
import torch.utils.data
from torch import nn
from torchvision.models import resnet50

from nncf.common.quantization.structs import QuantizerGroup
from nncf.torch import register_default_init_args
from nncf.torch.module_operations import UpdatePaddingValue
from nncf.torch.utils import get_all_modules_by_type
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.quantization.quantization_helpers import compare_multi_gpu_dump
from tests.torch.quantization.quantization_helpers import get_quantization_config_without_range_init
from tests.torch.quantization.test_hawq_precision_init import BaseConfigBuilder
from tests.torch.quantization.test_hawq_precision_init import check_bitwidth_graph
from tests.torch.quantization.test_hawq_precision_init import create_test_dataloaders
from tests.torch.quantization.test_hawq_precision_init import get_path_to_bitwidth_dump
from tests.torch.quantization.test_hawq_precision_init import precision_init_dumping_worker
from tests.torch.quantization.test_hawq_precision_init import ssd_vgg_512_test
from tests.torch.test_models import inception_v3
from tests.torch.test_models import squeezenet1_1
from tests.torch.test_models.mobilenet import mobilenet_v2


class AutoQConfigBuilder(BaseConfigBuilder):
    def __init__(
        self, config_creator_fn: Callable = None, batch_size=10, image_size=10, num_channels=3, num_init_samples=1
    ):
        super().__init__(config_creator_fn)
        if not config_creator_fn:
            self._config = self.create_autoq_test_config(
                batch_size, image_size, num_channels, num_init_samples=num_init_samples
            )
        self.for_npu()

    def eval_subset_ratio(self, eval_subset_ratio):
        self._options["eval_subset_ratio"] = str(eval_subset_ratio)
        self._config["compression"]["initializer"]["precision"]["eval_subset_ratio"] = eval_subset_ratio
        return self

    def iter_number(self, iter_number):
        self._options["iter_number"] = str(iter_number)
        self._config["compression"]["initializer"]["precision"]["iter_number"] = iter_number
        return self

    def warmup_iter_number(self, warmup_iter_number):
        self._options["warmup_iter_number"] = str(warmup_iter_number)
        self._config["compression"]["initializer"]["precision"]["warmup_iter_number"] = warmup_iter_number
        return self

    @staticmethod
    def create_autoq_test_config(batch_size=10, image_size=10, num_channels=3, num_init_samples=1):
        config = get_quantization_config_without_range_init()
        config["input_info"] = {
            "sample_size": [batch_size, num_channels, image_size, image_size],
        }
        config["batch_size"] = batch_size
        config["compression"].update(
            {
                "initializer": {
                    "precision": {
                        "type": "autoq",
                        "bits": [2, 4, 8],
                        "iter_number": 3,
                        "compression_ratio": 0.15,
                        "eval_subset_ratio": 1.0,
                        "warmup_iter_number": 2,
                    },
                    "range": {"num_init_samples": num_init_samples},
                    "batchnorm_adaptation": {"num_bn_adaptation_samples": 0},
                }
            }
        )
        return config


class AutoQTestStruct(NamedTuple):
    model_creator: Callable[[], nn.Module] = mobilenet_v2
    config_builder: AutoQConfigBuilder = AutoQConfigBuilder().for_npu()
    filename_suffix: str = "hw_config_npu"

    def __str__(self):
        return "_".join([self.model_creator.__name__, str(self.config_builder)])


RATIO = 0.4
AUTOQ_TEST_PARAMS = (
    AutoQTestStruct(config_builder=AutoQConfigBuilder()),
    AutoQTestStruct(config_builder=AutoQConfigBuilder().with_ratio(RATIO)),
    AutoQTestStruct(config_builder=AutoQConfigBuilder().with_ratio(RATIO).eval_subset_ratio(RATIO)),
    AutoQTestStruct(config_builder=AutoQConfigBuilder().eval_subset_ratio(RATIO)),
    AutoQTestStruct(
        model_creator=squeezenet1_1, config_builder=AutoQConfigBuilder().with_sample_size([1, 3, 224, 224])
    ),
    AutoQTestStruct(model_creator=resnet50, config_builder=AutoQConfigBuilder()),
    AutoQTestStruct(model_creator=resnet50, config_builder=AutoQConfigBuilder().iter_number(4).warmup_iter_number(2)),
    AutoQTestStruct(model_creator=resnet50, config_builder=AutoQConfigBuilder().with_ratio(RATIO)),
    AutoQTestStruct(model_creator=resnet50, config_builder=AutoQConfigBuilder().eval_subset_ratio(RATIO)),
    AutoQTestStruct(
        model_creator=resnet50, config_builder=AutoQConfigBuilder().with_ratio(RATIO).eval_subset_ratio(RATIO)
    ),
    AutoQTestStruct(
        model_creator=inception_v3,
        config_builder=AutoQConfigBuilder().with_sample_size([2, 3, 299, 299]).with_ratio(RATIO),
    ),
    AutoQTestStruct(
        model_creator=inception_v3,
        config_builder=AutoQConfigBuilder()
        .with_sample_size([2, 3, 299, 299])
        .with_ignored_scope(
            ["Inception3/BasicConv2d[Conv2d_2a_3x3]/NNCFConv2d[conv]/conv2d_0"], target_group=QuantizerGroup.WEIGHTS
        )
        .eval_subset_ratio(RATIO),
    ),
    AutoQTestStruct(
        model_creator=ssd_vgg_512_test,
        config_builder=AutoQConfigBuilder().with_sample_size([1, 3, 512, 512]).eval_subset_ratio(RATIO),
    ),
    AutoQTestStruct(
        model_creator=ssd_vgg_512_test,
        config_builder=AutoQConfigBuilder().with_sample_size([1, 3, 512, 512]).with_ratio(RATIO),
    ),
)


@pytest.mark.cuda
@pytest.mark.parametrize("params", AUTOQ_TEST_PARAMS, ids=[str(p) for p in AUTOQ_TEST_PARAMS])
def test_autoq_precision_init(_seed, dataset_dir, tmp_path, mocker, params):
    config = params.config_builder.build()
    model = params.model_creator()
    if torch.cuda.is_available():
        model = model.cuda()

    config["log_dir"] = str(tmp_path)

    if not dataset_dir:
        dataset_dir = str(tmp_path)
    train_loader, _ = create_test_dataloaders(config, dataset_dir)

    from nncf.torch.automl.agent.ddpg.ddpg import DDPG

    random_action_spy = mocker.spy(DDPG, "random_action")
    select_action_spy = mocker.spy(DDPG, "select_action")

    from nncf.torch.quantization.precision_init.autoq_init import AutoQPrecisionInitializer

    autoq_obj_init_spy = mocker.spy(AutoQPrecisionInitializer, "__init__")
    adjust_pad_creation_spy = mocker.spy(UpdatePaddingValue, "__init__")

    config = register_default_init_args(
        config, train_loader, autoq_eval_fn=lambda *x: random(), val_loader=train_loader
    )
    model, algo_ctrl = create_compressed_model_and_algo_for_test(model, config)

    bw_init_config = config["compression"]["initializer"]["precision"]
    learning_iter_number = bw_init_config["iter_number"] - bw_init_config["warmup_iter_number"]

    experimental_ctrl = autoq_obj_init_spy.call_args[0][1]
    n_quantizer = len(experimental_ctrl.all_quantizations)

    assert random_action_spy.call_count == bw_init_config["warmup_iter_number"] * n_quantizer
    assert (
        select_action_spy.call_count == learning_iter_number * (n_quantizer + 1) + bw_init_config["warmup_iter_number"]
    )
    final_num_of_adjust_pad_ops = len(get_all_modules_by_type(model, "UpdatePaddingValue"))
    assert adjust_pad_creation_spy.call_count == final_num_of_adjust_pad_ops

    path_to_dot = "{}_{}.dot".format(params.model_creator.__name__, params.config_builder.filename_suffix())
    graph_dir = os.path.join("quantized", "autoq")
    check_bitwidth_graph(algo_ctrl, model, path_to_dot, graph_dir)


@pytest.mark.cuda
def test_can_broadcast_initialized_precisions_in_distributed_mode(tmp_path, runs_subprocess_in_precommit):
    if not torch.cuda.is_available():
        pytest.skip("Skipping CUDA test cases for CPU only setups")
    config_builder = AutoQConfigBuilder(batch_size=2).for_trial()
    config = config_builder.build()
    ngpus_per_node = torch.cuda.device_count()
    config.world_size = ngpus_per_node

    torch.multiprocessing.spawn(
        precision_init_dumping_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, tmp_path), join=True
    )

    assert not compare_multi_gpu_dump(config, tmp_path, get_path_to_bitwidth_dump)
