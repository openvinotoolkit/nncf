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
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Union

import jstyleson as json
import pytest
import torch.cuda
from packaging import version
from pytest import approx

from tests.cross_fw.shared.paths import PROJECT_ROOT
from tests.torch.helpers import Command
from tests.torch.sample_test_validator import BaseSampleTestCaseDescriptor
from tests.torch.sample_test_validator import BaseSampleValidator
from tests.torch.sparsity.movement.helpers import LINEAR_LAYER_SPARSITY_NAME_IN_MOVEMENT_STAT
from tests.torch.sparsity.movement.helpers import MRPC_CONFIG_FILE_NAME
from tests.torch.sparsity.movement.helpers import TRAINING_SCRIPTS_PATH


class MovementGlueHandler:
    def get_executable(self) -> Path:
        return TRAINING_SCRIPTS_PATH.joinpath(self._get_main_filename() + ".py")

    @staticmethod
    def get_checkpoint_path(checkpoint_save_dir) -> Path:
        return Path(checkpoint_save_dir)

    def get_main_location(self) -> str:
        return ".".join(["tests", "torch", "sparsity", "movement", "training_scripts", self._get_main_filename()])

    def _get_main_filename(self) -> str:
        return "run_glue"

    def get_metric_value_from_checkpoint(self, checkpoint_save_dir: str) -> Dict[str, Union[float, int]]:
        checkpoint_path = self.get_checkpoint_path(checkpoint_save_dir)
        result_path = checkpoint_path / "all_results.json"
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        return result


class MovementTrainingTestDescriptor(BaseSampleTestCaseDescriptor):
    def __init__(self):
        super().__init__()
        self.sample_type_ = None
        self.sample_handler = MovementGlueHandler()
        self.model_name_ = "google/bert_uncased_L-2_H-128_A-2"
        self.enable_autocast_fp16_ = False
        self.distributed_data_parallel_ = False
        self.n_card = 1
        self.cpu_only_ = False
        self.execution_arg = "single_card"
        self.timeout_ = 8 * 60  # 8 mins
        self.expected_eval_acc_ = None
        self.expected_eval_f1_ = None
        self.expected_linear_layer_sparsity_ = None
        self.num_train_epochs_ = 9
        self.learning_rate_ = 5e-5
        self.seed_ = None
        self.output_dir = None
        self.quick_check_ = False

    def finalize(self, dataset_dir, tmp_path_factory, weekly_models_path):
        config_name = Path(self.config_name_).stem
        if weekly_models_path is None or self.quick_check_:
            config_name += "_quick-check"
        is_fp16_str = "autocast-fp16" if self.enable_autocast_fp16_ else "fp32training"
        self.output_dir = tmp_path_factory.mktemp("models") / Path(self.execution_arg, config_name, is_fp16_str)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        return self

    @property
    def config_directory(self) -> Path:
        return TRAINING_SCRIPTS_PATH

    def model_name(self, model_name_):
        self.model_name_ = model_name_
        return self

    def timeout_seconds(self, num_seconds: int):
        self.timeout_ = num_seconds
        return self

    def expected_eval_acc(self, expected):
        self.expected_eval_acc_ = expected
        return self

    def expected_eval_f1(self, expected):
        self.expected_eval_f1_ = expected
        return self

    def expected_linear_layer_sparsity(self, expected):
        self.expected_linear_layer_sparsity_ = expected
        return self

    def num_train_epochs(self, num_train_epochs: int):
        self.num_train_epochs_ = num_train_epochs
        return self

    def learning_rate(self, learning_rate: float):
        self.learning_rate_ = learning_rate
        return self

    def seed(self, seed_):
        self.seed_ = seed_
        return self

    def distributed_data_parallel(self, distributed_data_parallel_=True, n_card=2):
        if distributed_data_parallel_ is True:
            assert self.cpu_only_ is False
        self.execution_arg = "multiprocessing-distributed"
        self.distributed_data_parallel_ = distributed_data_parallel_
        self.n_card = n_card
        return self

    def cpu_only(self, cpu_only_=True):
        if cpu_only_:
            self.execution_arg = "cpu-only"
        self.cpu_only_ = cpu_only_
        return self

    def data_parallel(self, data_parallel_=True, n_card=2):
        if data_parallel_:
            self.execution_arg = "data-parallel"
        self.n_card = n_card
        return self

    def enable_autocast_fp16(self, enable_autocast_fp16_: bool = True):
        self.enable_autocast_fp16_ = enable_autocast_fp16_
        return self

    def quick_check(self, quick_check_: bool = True):
        self.quick_check_ = quick_check_
        return self

    def get_validator(self):
        return MovementTrainingValidator(self)

    def get_metric(self) -> Dict[str, float]:
        return self.sample_handler.get_metric_value_from_checkpoint(self.output_dir)

    def __str__(self):
        return "_".join([self.config_name_, self.dataset_name, self.execution_arg])


class MovementTrainingValidator(BaseSampleValidator):
    def __init__(self, desc: MovementTrainingTestDescriptor):
        self._desc = desc
        self._sample_handler = desc.sample_handler

    def validate_sample(self, args, mocker):
        cmd = self._create_command_line(args)
        env_with_cuda_reproducibility = os.environ.copy()
        env_with_cuda_reproducibility["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env_with_cuda_reproducibility["PYTHONPATH"] = str(PROJECT_ROOT)
        if not self._desc.cpu_only_:
            CUDA_ENV_KEY = "CUDA_VISIBLE_DEVICES"
            n_card = self._desc.n_card
            if CUDA_ENV_KEY not in os.environ:
                dev_ids = [str(i) for i in range(n_card)]
            else:
                all_dev_ids = os.environ[CUDA_ENV_KEY].split(",")
                dev_ids = all_dev_ids[:n_card]
            env_with_cuda_reproducibility[CUDA_ENV_KEY] = ",".join(dev_ids)
        runner = Command(cmd, env=env_with_cuda_reproducibility)
        runner.run(timeout=self._desc.timeout_)

    def get_default_args(self):
        args = {
            "model_name_or_path": self._desc.model_name_,
            "task_name": self._desc.dataset_name,
            "nncf_config": self._desc.config_path,
            "do_train": True,
            "do_eval": True,
            "num_train_epochs": self._desc.num_train_epochs_,
            "evaluation_strategy": "epoch",
            "output_dir": self._desc.output_dir,
            "seed": self._desc.seed_,
            "learning_rate": self._desc.learning_rate_,
            "per_device_train_batch_size": self._desc.batch_size_,
        }
        if self._desc.enable_autocast_fp16_:
            args["fp16"] = True
        if self._desc.cpu_only_:
            args["no_cuda"] = True
        if self._desc.quick_check_:
            args["quick_check"] = True
        return args

    def _create_command_line(self, args):
        main_py = self._sample_handler.get_executable()
        cli_args_l = []
        for key, val in args.items():
            key = f"--{key}"
            if val in [None, True]:
                cli_args_l.append(key)
            elif (not isinstance(val, bool)) or val is not False:
                cli_args_l.extend([key, val])
        cli_args = " ".join(map(str, cli_args_l))
        extra_for_ddp = ""
        if self._desc.distributed_data_parallel_:
            extra_for_ddp = f"-m torch.distributed.run --nproc_per_node={self._desc.n_card}"
        return f"{sys.executable} {extra_for_ddp} {main_py} {cli_args}"

    def setup_spy(self, mocker):
        pass

    def validate_spy(self):
        pass


mrpc_movement_desc_template = (
    MovementTrainingTestDescriptor()
    .model_name("google/bert_uncased_L-2_H-128_A-2")
    .real_dataset("mrpc")
    .config_name(MRPC_CONFIG_FILE_NAME)
    .learning_rate(5e-5)
    .batch_size(64)
    .num_train_epochs(9)
    .seed(42)
)

MOVEMENT_DESCRIPTORS = {
    "mrpc_cuda_1card": deepcopy(mrpc_movement_desc_template)
    .expected_eval_f1(approx(0.81, abs=0.02))
    .expected_eval_acc(approx(0.68, abs=0.03))
    .expected_linear_layer_sparsity(approx(0.48, abs=0.05)),
    "mrpc_cuda_1card_fp16": deepcopy(mrpc_movement_desc_template)
    .enable_autocast_fp16()
    .expected_eval_f1(approx(0.81, abs=0.02))
    .expected_eval_acc(approx(0.68, abs=0.03))
    .expected_linear_layer_sparsity(approx(0.48, abs=0.05)),
    "mrpc_cuda_2cards_dp": deepcopy(mrpc_movement_desc_template)
    .batch_size(32)
    .data_parallel(n_card=2)
    .expected_eval_f1(approx(0.81, abs=0.02))
    .expected_eval_acc(approx(0.68, abs=0.03))
    .expected_linear_layer_sparsity(approx(0.48, abs=0.05)),
    "mrpc_cuda_2cards_dp_fp16": deepcopy(mrpc_movement_desc_template)
    .batch_size(32)
    .data_parallel(n_card=2)
    .enable_autocast_fp16()
    .expected_eval_f1(approx(0.81, abs=0.02))
    .expected_eval_acc(approx(0.68, abs=0.03))
    .expected_linear_layer_sparsity(approx(0.48, abs=0.05)),
    "mrpc_cuda_2cards_ddp": deepcopy(mrpc_movement_desc_template)
    .batch_size(32)
    .distributed_data_parallel(n_card=2)
    .expected_eval_f1(approx(0.81, abs=0.02))
    .expected_eval_acc(approx(0.68, abs=0.03))
    .expected_linear_layer_sparsity(approx(0.48, abs=0.05)),
    "mrpc_cuda_2cards_ddp_fp16": deepcopy(mrpc_movement_desc_template)
    .batch_size(32)
    .distributed_data_parallel(n_card=2)
    .enable_autocast_fp16()
    .expected_eval_f1(approx(0.81, abs=0.02))
    .expected_eval_acc(approx(0.68, abs=0.03))
    .expected_linear_layer_sparsity(approx(0.48, abs=0.05)),
    "mrpc_cpu_1card": deepcopy(mrpc_movement_desc_template)
    .cpu_only()
    .expected_eval_f1(approx(0.81, abs=0.02))
    .expected_eval_acc(approx(0.68, abs=0.03))
    .expected_linear_layer_sparsity(approx(0.48, abs=0.05)),
}


def finalize_desc(
    desc: MovementTrainingTestDescriptor, is_long_training: bool, dataset_dir, tmp_path_factory, weekly_models_path
):
    if is_long_training and (weekly_models_path is None):
        pytest.skip("Skip the test for long training since `--weekly-models` option is not specified.")
    if (not is_long_training) and (weekly_models_path is not None):
        pytest.skip("Skip the test for short training since a long run will be checked.")
    if desc.cpu_only_ or (desc.n_card > 1 and not desc.distributed_data_parallel_):
        pytest.xfail("Training gets stuck in pytest, but is working fine if launched in normal bash.")
    return desc.finalize(dataset_dir, tmp_path_factory, weekly_models_path)


@pytest.fixture(
    name="movement_desc_long", scope="module", params=MOVEMENT_DESCRIPTORS.values(), ids=MOVEMENT_DESCRIPTORS.keys()
)
def fixture_movement_desc_long(request, dataset_dir, tmp_path_factory, weekly_models_path):
    desc: MovementTrainingTestDescriptor = request.param
    return finalize_desc(desc, True, dataset_dir, tmp_path_factory, weekly_models_path)


@pytest.fixture(
    name="movement_desc_short", scope="module", params=MOVEMENT_DESCRIPTORS.values(), ids=MOVEMENT_DESCRIPTORS.keys()
)
def fixture_movement_desc_short(request, dataset_dir, tmp_path_factory, weekly_models_path):
    desc: MovementTrainingTestDescriptor = request.param
    desc = deepcopy(desc).quick_check()
    return finalize_desc(desc, False, dataset_dir, tmp_path_factory, weekly_models_path)


class TestMovementTraining:
    @pytest.mark.weekly
    def test_compression_movement_long_train(self, movement_desc_long: MovementTrainingTestDescriptor, mocker):
        if (not movement_desc_long.cpu_only_) and torch.cuda.device_count() < movement_desc_long.n_card:
            pytest.skip(f"No enough cuda devices to run {movement_desc_long}")
        validator = movement_desc_long.get_validator()
        args = validator.get_default_args()
        validator.validate_sample(args, mocker)
        self._validate_model_is_saved(movement_desc_long)
        self._validate_train_metric(movement_desc_long)

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("1.12"),
        reason=f"torch {torch.__version__} may not compatible with installed transformers package. "
        f"Some tests may fail with error",
    )
    @pytest.mark.nightly
    def test_compression_movement_short_train(self, movement_desc_short: MovementTrainingTestDescriptor, mocker):
        if (not movement_desc_short.cpu_only_) and torch.cuda.device_count() < movement_desc_short.n_card:
            pytest.skip(f"No enough cuda devices to run {movement_desc_short}")
        validator = movement_desc_short.get_validator()
        args = validator.get_default_args()
        validator.validate_sample(args, mocker)
        self._validate_model_is_saved(movement_desc_short)

    @staticmethod
    def _validate_model_is_saved(desc: MovementTrainingTestDescriptor):
        assert Path(desc.output_dir, "model.safetensors").is_file()

    @staticmethod
    def _validate_train_metric(desc: MovementTrainingTestDescriptor):
        metrics = desc.get_metric()
        if desc.expected_eval_acc_ is not None:
            assert metrics["eval_accuracy"] == approx(desc.expected_eval_acc_)
        if desc.expected_eval_f1_ is not None:
            assert metrics["eval_f1"] == approx(desc.expected_eval_f1_)
        if desc.expected_linear_layer_sparsity_ is not None:
            assert metrics[LINEAR_LAYER_SPARSITY_NAME_IN_MOVEMENT_STAT] == approx(desc.expected_linear_layer_sparsity_)
