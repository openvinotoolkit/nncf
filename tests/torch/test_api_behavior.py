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

import pytest
import torch
from torch.utils.data import DataLoader

import nncf
from nncf import NNCFConfig
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.torch import create_compressed_model
from nncf.torch import register_default_init_args
from nncf.torch.tensor_statistics.algo import TensorStatisticsCollectionBuilder
from nncf.torch.tensor_statistics.algo import TensorStatisticsCollectionController
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import OnesDatasetMock
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.nncf_network.helpers import SimplestModel

INPUT_SAMPLE_SIZE = [1, 1, 4, 4]
CONFIG_WITH_ALL_INIT_TYPES = {
    "model": "basic_quant_conv",
    "input_info": {
        "sample_size": INPUT_SAMPLE_SIZE,
    },
    "compression": {
        "algorithm": "quantization",
        "initializer": {
            "precision": {"type": "hawq", "bits": [4, 8, 6], "num_data_points": 1, "iter_number": 1, "tolerance": 1e-2},
            "range": {"num_init_samples": 1},
            "batchnorm_adaptation": {"num_bn_adaptation_samples": 5},
        },
    },
}


@pytest.fixture(name="nncf_config_with_default_init_args")
def nncf_config_with_default_init_args_(mocker):
    config = NNCFConfig.from_dict(CONFIG_WITH_ALL_INIT_TYPES)

    train_loader = DataLoader(
        OnesDatasetMock(INPUT_SAMPLE_SIZE[1:]),
        batch_size=1,
        num_workers=0,  # Workaround for PyTorch MultiprocessingDataLoader issues
        shuffle=False,
    )
    mocker_criterion = mocker.stub()
    mocker_criterion.batch_size = 1

    config = register_default_init_args(config, train_loader, criterion=mocker_criterion)
    return config


@pytest.mark.parametrize(
    ("config_cutter", "tensor_statistics_collection_count", "precision_init_call_count", "bn_adaptation_call_count"),
    [
        # 1 stat collection for setting up an experimental quantization setup for precision init,
        # + 1 stat collection for implicit range initialization with default parameters
        (lambda x: x["initializer"].pop("range"), 2, 1, 1),
        (lambda x: x.pop("initializer"), 1, 0, 1),
        (lambda x: x["initializer"].pop("precision"), 1, 0, 1),
        (lambda x: x["initializer"]["range"].update({"num_init_samples": 0}), 0, 1, 1),
    ],
    ids=["precision_init_only", "no_init_params", "range_init_only", "skip_range_init"],
)
def test_range_init_is_called(
    nncf_config_with_default_init_args,
    config_cutter,
    tensor_statistics_collection_count,
    precision_init_call_count,
    bn_adaptation_call_count,
    mocker,
):
    config = nncf_config_with_default_init_args
    model = BasicConvTestModel()

    _ = mocker.patch("nncf.torch.initialization.SimpleDataLoaderRunner.run")
    stat_builder_apply_to_spy = mocker.spy(TensorStatisticsCollectionBuilder, "apply_to")
    stat_builder_build_controller_mm = mocker.patch(
        "nncf.torch.tensor_statistics.algo.TensorStatisticsCollectionBuilder.build_controller"
    )
    stat_builder_build_controller_mm.return_value = TensorStatisticsCollectionController(None, {})

    precision_init_spy = mocker.patch(
        "nncf.torch.quantization.precision_init.hawq_init.HAWQPrecisionInitializer.apply_init", autospec=True
    )  # autospec=True will patch the function as an instance method
    bn_adaptation_spy = mocker.patch("nncf.torch.initialization.DataLoaderBNAdaptationRunner.run")

    def fn(self) -> SingleConfigQuantizerSetup:
        return self._algo.get_quantizer_setup_for_current_state()

    precision_init_spy.side_effect = fn

    config_cutter(config["compression"])
    create_compressed_model_and_algo_for_test(model, config)

    assert stat_builder_apply_to_spy.call_count == tensor_statistics_collection_count
    assert precision_init_spy.call_count == precision_init_call_count
    assert bn_adaptation_spy.call_count == bn_adaptation_call_count


class DeviceCheckingModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = TwoConvTestModel()
        self.original_device = device
        self.model.to(device)

    def forward(self, x):
        for param in self.model.parameters():
            # 'in' to handle the situation when .to('cuda') results in 'cuda:0' actual device
            assert self.original_device in str(param.device)
        return self.model.forward(x)


@pytest.mark.parametrize(
    "original_device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.cuda), pytest.param("cuda:0", marks=pytest.mark.cuda)],
)
def test_model_is_inited_with_own_device_by_default(nncf_config_with_default_init_args, original_device):
    if not torch.cuda.is_available() and "cuda" in original_device:
        pytest.skip("Skipping for CPU-only setups")
    model = DeviceCheckingModel(original_device)
    create_compressed_model_and_algo_for_test(model, nncf_config_with_default_init_args)


def test_repeat_compression_fails():
    model = SimplestModel()
    nncf_config = NNCFConfig.from_dict({"input_info": {"sample_size": SimplestModel.INPUT_SIZE}})
    _ = create_compressed_model(model, nncf_config)
    with pytest.raises(nncf.InternalError, match="The model object has already been compressed."):
        _ = create_compressed_model(model, nncf_config)
