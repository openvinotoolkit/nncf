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
import torch
from torch import nn
from torch.utils.data import DataLoader

from nncf.common.statistics import NNCFStatistics
from nncf.config.structures import QuantizationRangeInitArgs
from nncf.torch import register_default_init_args
from nncf.torch.dynamic_graph.io_handling import FillerInputInfo
from nncf.torch.initialization import wrap_dataloader_for_init
from nncf.torch.quantization.base_ctrl import QuantizationControllerBase
from nncf.torch.quantization.schedulers import StagedQuantizationScheduler
from tests.torch.helpers import OnesDatasetMock
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.quantization.test_algo_quantization import get_squeezenet_quantization_config
from tests.torch.test_models import squeezenet1_1


def create_staged_scheduler(ctrl_spy, w_start=2, a_start=1):
    params = {"activations_quant_start_epoch": a_start, "weights_quant_start_epoch": w_start}
    scheduler = StagedQuantizationScheduler(ctrl_spy.get_mocked_algo(), params)
    return scheduler


class QuantizationControllerBaseForTest(QuantizationControllerBase):
    @property
    def loss(self):
        pass

    @property
    def scheduler(self):
        pass

    def statistics(self, quickly_collected_only: bool = False):
        return NNCFStatistics()

    def compression_stage(self):
        pass


class QuantizationCtrlBaseSpy:
    def __init__(self, mocker):
        self._mocked_ctrl = QuantizationControllerBaseForTest(mocker.stub)
        mocker.patch.object(self._mocked_ctrl, "enable_weight_quantization")
        mocker.patch.object(self._mocked_ctrl, "enable_activation_quantization")
        mocker.patch.object(self._mocked_ctrl, "disable_weight_quantization")
        mocker.patch.object(self._mocked_ctrl, "disable_activation_quantization")
        mocker.patch.object(self._mocked_ctrl, "init_range")

    def enable_weight_count(self):
        return self._mocked_ctrl.enable_weight_quantization.call_count

    def enable_activation_count(self):
        return self._mocked_ctrl.enable_activation_quantization.call_count

    def disable_weight_count(self):
        return self._mocked_ctrl.disable_weight_quantization.call_count

    def disable_activation_count(self):
        return self._mocked_ctrl.disable_activation_quantization.call_count

    def init_range_count(self):
        return self._mocked_ctrl.init_range.call_count

    def get_mocked_algo(self):
        return self._mocked_ctrl

    def check_call_counts(
        self,
        enable_weight_count: int,
        enable_activation_count: int,
        disable_weight_count: int,
        disable_activation_count: int,
        init_range_count: int,
    ):
        assert self.enable_weight_count() == enable_weight_count, "enable weight count mismatch"
        assert self.enable_activation_count() == enable_activation_count, "enable activation count mismatch"
        assert self.disable_weight_count() == disable_weight_count, "disable weight count mismatch"
        assert self.disable_activation_count() == disable_activation_count, "disable activation count mismatch"
        assert self.init_range_count() == init_range_count, "init range count mismatch"


def test_scheduler_not_enables_quantizations__by_default(mocker):
    ctrl_spy = QuantizationCtrlBaseSpy(mocker)
    StagedQuantizationScheduler(ctrl_spy.get_mocked_algo())
    ctrl_spy.check_call_counts(0, 0, 1, 1, 0)


def test_staged_scheduler_enables_quantizations__with_zero(mocker):
    ctrl_spy = QuantizationCtrlBaseSpy(mocker)
    create_staged_scheduler(ctrl_spy, 0, 0)
    ctrl_spy.check_call_counts(1, 1, 0, 0, 0)


def test_staged_scheduler_enables_quantizations_on_epoch_step(mocker):
    ctrl_spy = QuantizationCtrlBaseSpy(mocker)
    scheduler = create_staged_scheduler(ctrl_spy)
    ctrl_spy.check_call_counts(0, 0, 1, 1, 0)

    scheduler.epoch_step()
    ctrl_spy.check_call_counts(0, 0, 1, 1, 0)

    scheduler.epoch_step()
    ctrl_spy.check_call_counts(0, 1, 1, 1, 1)

    scheduler.epoch_step()
    ctrl_spy.check_call_counts(1, 1, 1, 1, 2)


def test_staged_scheduler_enables_quantizations_on_epoch_step__at_the_same_time(mocker):
    ctrl_spy = QuantizationCtrlBaseSpy(mocker)
    scheduler = create_staged_scheduler(ctrl_spy, 1, 1)
    ctrl_spy.check_call_counts(0, 0, 1, 1, 0)

    scheduler.epoch_step()
    ctrl_spy.check_call_counts(0, 0, 1, 1, 0)

    scheduler.epoch_step()
    ctrl_spy.check_call_counts(1, 1, 1, 1, 1)


def test_staged_scheduler_enables_quantizations_on_load(mocker):
    old_ctrl_spy = QuantizationCtrlBaseSpy(mocker)
    old_scheduler = create_staged_scheduler(old_ctrl_spy)
    old_scheduler.epoch_step()
    old_scheduler.epoch_step()
    old_scheduler.epoch_step()
    scheduler_state = old_scheduler.get_state()

    ctrl_spy = QuantizationCtrlBaseSpy(mocker)
    scheduler = create_staged_scheduler(ctrl_spy, 1, 3)
    ctrl_spy.check_call_counts(0, 0, 1, 1, 0)

    scheduler.load_state(scheduler_state)
    ctrl_spy.check_call_counts(1, 0, 1, 2, 0)

    scheduler.epoch_step()
    ctrl_spy.check_call_counts(1, 1, 1, 2, 1)


def test_staged_scheduler_with_empty_quantization():
    config = get_squeezenet_quantization_config()
    config["compression"].update(
        {
            "params": {
                "activations_quant_start_epoch": 1,
                "weights_quant_start_epoch": 2,
            }
        }
    )
    register_bn_adaptation_init_args(config)
    model = squeezenet1_1(num_classes=10, dropout=0)

    model, algo = create_compressed_model_and_algo_for_test(model, config)
    scheduler = algo.scheduler
    for module in algo.all_quantizations.values():
        assert not module.is_enabled_quantization()

    scheduler.epoch_step()
    for module in algo.all_quantizations.values():
        assert not module.is_enabled_quantization()
    scheduler.epoch_step()
    for wq_info in algo.weight_quantizers.values():
        assert not wq_info.quantizer_module_ref.is_enabled_quantization()
    for aq_info in algo.non_weight_quantizers.values():
        assert aq_info.quantizer_module_ref.is_enabled_quantization()

    scheduler.epoch_step()
    for module in algo.all_quantizations.values():
        assert module.is_enabled_quantization()


def test_staged_scheduler_with_range_init():
    config = get_squeezenet_quantization_config()
    config["compression"].update(
        {
            "params": {
                "activations_quant_start_epoch": 1,
                "weights_quant_start_epoch": 2,
            },
            "initializer": {"range": {"num_init_samples": 1}},
        }
    )
    register_bn_adaptation_init_args(config)
    model = squeezenet1_1(num_classes=10, dropout=0)

    input_infos_list = FillerInputInfo.from_nncf_config(config)
    input_sample_size = input_infos_list.elements[0].shape
    data_loader = DataLoader(
        OnesDatasetMock(input_sample_size[1:]),
        batch_size=1,
        num_workers=0,  # Workaround for PyTorch MultiprocessingDataLoader issues
        shuffle=False,
    )
    config.register_extra_structs([QuantizationRangeInitArgs(wrap_dataloader_for_init(data_loader))])

    model, algo = create_compressed_model_and_algo_for_test(model, config)
    scheduler = algo.scheduler

    for module in algo.all_quantizations.values():
        assert not module.is_enabled_quantization()

    scheduler.epoch_step()
    for module in algo.all_quantizations.values():
        assert not module.is_enabled_quantization()

    scheduler.epoch_step()

    for wq_info in algo.weight_quantizers.values():
        assert not wq_info.quantizer_module_ref.is_enabled_quantization()
    for aq_info in algo.non_weight_quantizers.values():
        assert aq_info.quantizer_module_ref.is_enabled_quantization()

    scheduler.epoch_step()
    for module in algo.all_quantizations.values():
        assert module.is_enabled_quantization()


class HawqDatasetMock:
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes
        super().__init__()

    def __getitem__(self, index):
        return torch.ones(self.input_size), torch.LongTensor([1]).squeeze_()

    def __len__(self):
        return 1


def test_staged_scheduler_with_hawq():
    config = get_squeezenet_quantization_config()
    config["compression"].update(
        {
            "params": {
                "activations_quant_start_epoch": 1,
                "weights_quant_start_epoch": 2,
            },
            "initializer": {
                "range": {"num_init_samples": 1},
                "precision": {"type": "hawq", "num_data_points": 1, "iter_number": 1, "tolerance": 1},
            },
        }
    )
    num_classes = 10
    model = squeezenet1_1(num_classes=num_classes, dropout=0)

    input_infos_list = FillerInputInfo.from_nncf_config(config)
    input_sample_size = input_infos_list.elements[0].shape
    data_loader = DataLoader(
        HawqDatasetMock(input_sample_size[1:], num_classes),
        batch_size=1,
        num_workers=0,  # Workaround for PyTorch MultiprocessingDataLoader issues
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss().cuda()
    config = register_default_init_args(config, data_loader, criterion=criterion)

    model, algo = create_compressed_model_and_algo_for_test(model, config)
    scheduler = algo.scheduler

    for module in algo.all_quantizations.values():
        assert not module.is_enabled_quantization()

    scheduler.epoch_step()
    for module in algo.all_quantizations.values():
        assert not module.is_enabled_quantization()

    scheduler.epoch_step()
    for wq_info in algo.weight_quantizers.values():
        assert not wq_info.quantizer_module_ref.is_enabled_quantization()
    for aq_info in algo.non_weight_quantizers.values():
        assert aq_info.quantizer_module_ref.is_enabled_quantization()

    scheduler.epoch_step()
    for module in algo.all_quantizations.values():
        assert module.is_enabled_quantization()
