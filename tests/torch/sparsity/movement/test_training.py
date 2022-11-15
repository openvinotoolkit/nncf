import sys
import os
from typing import Dict, Union
from pathlib import Path
import pytest
from pytest import approx
import jstyleson as json
from copy import deepcopy

import torch.cuda
from tests.common.helpers import PROJECT_ROOT
from tests.common.helpers import TEST_ROOT
from tests.torch.helpers import Command
from tests.torch.sample_test_validator import BaseSampleTestCaseDescriptor
from tests.torch.sample_test_validator import BaseSampleValidator


class MovementGlueHandler:
    def get_executable(self) -> Path:
        return TEST_ROOT.joinpath("torch", "sparsity", "movement", "examples", self._get_main_filename() + ".py")

    @staticmethod
    def get_checkpoint_path(checkpoint_save_dir):
        return Path(checkpoint_save_dir)

    def get_main_location(self) -> str:
        return ".".join(["tests", "torch", "sparsity", "movement", "examples", self._get_main_filename()])

    def _get_main_filename(self):
        return "run_glue"

    def get_metric_value_from_checkpoint(self, checkpoint_save_dir: str) -> Dict[str, Union[float, int]]:
        checkpoint_path = self.get_checkpoint_path(checkpoint_save_dir)
        state_path = checkpoint_path / "trainer_state.json"
        with open(state_path, "r") as f:
            state_dict = json.load(f)
        max_step = max(log["step"] for log in state_dict["log_history"])

        # gather useful info in one dict
        result = dict(step=max_step)
        for log in state_dict["log_history"]:
            if log["step"] == max_step:
                result.update(log)
        return result


class MovementTrainingTestDescriptor(BaseSampleTestCaseDescriptor):
    def __init__(self):
        super().__init__()
        self.sample_type_ = None
        self.sample_handler = MovementGlueHandler()
        self.model_name_ = "google/bert_uncased_L-2_H-128_A-2"
        self.enable_autocast_fp16_ = False
        self.distributed_data_parallel_ = False
        self.n_process = 1
        self.cpu_only_ = False
        self.execution_arg = "single_card"
        self.timeout_ = 10 * 60  # 10 mins
        self.expected_eval_acc_ = None
        self.expected_eval_f1_ = None
        self.expected_rela_sparsity_ = None
        self.num_train_epochs_ = 6
        self.learning_rate_ = 1e-4
        self.seed_ = None
        self.output_dir = None
        self.quick_check_ = False

    def finalize(self, dataset_dir, tmp_path_factory, weekly_models_path):
        config_name = Path(self.config_name_).stem
        if weekly_models_path is None or self.quick_check_:
            config_name += '_quick-check'
        is_fp16_str = "autocast-fp16" if self.enable_autocast_fp16_ else "fp32training"
        self.output_dir = tmp_path_factory.mktemp("models") / Path(
            self.execution_arg, config_name, is_fp16_str
        )
        self.output_dir.mkdir(exist_ok=True, parents=True)
        return self

    @property
    def config_directory(self) -> Path:
        return TEST_ROOT.joinpath("torch", "sparsity", "movement", "examples")

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

    def expected_rela_sparsity(self, expected):
        self.expected_rela_sparsity_ = expected
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

    def distributed_data_parallel(self, distributed_data_parallel_=True, n_process=2):
        if distributed_data_parallel_ is True:
            assert self.cpu_only_ is False
        self.execution_arg = "multiprocessing-distributed"
        self.distributed_data_parallel_ = distributed_data_parallel_
        self.n_process = n_process
        return self

    def cpu_only(self, cpu_only_=True):
        if cpu_only_:
            self.execution_arg = "cpu-only"
        self.cpu_only_ = cpu_only_
        return self

    def data_parallel(self, data_parallel_=True, n_process=2):
        if data_parallel_:
            self.execution_arg = "data-parallel"
        self.n_process = n_process
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
        runner = Command(cmd)
        env_with_cuda_reproducibility = os.environ.copy()
        env_with_cuda_reproducibility["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if not self._desc.cpu_only_:
            dev_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            n_process = self._desc.n_process
            env_with_cuda_reproducibility["CUDA_VISIBLE_DEVICES"] = ",".join(
                dev_ids[:n_process])
        runner.kwargs.update(env=env_with_cuda_reproducibility)
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
        project_root = PROJECT_ROOT.as_posix()
        main_py = self._sample_handler.get_executable()
        cli_args_l = []
        for key, val in args.items():
            key = f'--{key}'
            if val in [None, True]:
                cli_args_l.append(key)
            elif (not isinstance(val, bool)) or val is not False:
                cli_args_l.extend([key, val])
        cli_args = ' '.join(map(str, cli_args_l))
        extra_for_ddp = ""
        if self._desc.distributed_data_parallel_:
            extra_for_ddp = f"-m torch.distributed.run --nproc_per_node={self._desc.n_process}"
        return f"PYTHONPATH={project_root} {sys.executable} {extra_for_ddp} {main_py} {cli_args}"

    def setup_spy(self, mocker):
        pass

    def validate_spy(self):
        pass

mrpc_movement_desc_template = \
    MovementTrainingTestDescriptor()\
    .model_name("google/bert_uncased_L-2_H-128_A-2")\
    .real_dataset("mrpc")\
    .config_name("bert_tiny_uncased_mrpc_movement.json")\
    .learning_rate(1e-4)\
    .batch_size(128)\
    .num_train_epochs(5)\
    .seed(42)\

MOVEMENT_DESCRIPTORS = {
    # TODO(yujie): update expected metrics
    "mrpc_cuda_1proc": deepcopy(mrpc_movement_desc_template)
    .expected_eval_acc(approx(0.5, abs=0.5))
    .expected_rela_sparsity(approx(0.5, abs=0.5)),

    "mrpc_cuda_1proc_fp16": deepcopy(mrpc_movement_desc_template)
    .enable_autocast_fp16()
    .expected_eval_acc(approx(0.5, abs=0.5))
    .expected_rela_sparsity(approx(0.5, abs=0.5)),

    "mrpc_cuda_2proc_dp": deepcopy(mrpc_movement_desc_template)
    .batch_size(64)
    .data_parallel(n_process=2)
    .expected_eval_acc(approx(0.5, abs=0.5))
    .expected_rela_sparsity(approx(0.5, abs=0.5)),

    "mrpc_cuda_2proc_dp_fp16": deepcopy(mrpc_movement_desc_template)
    .batch_size(64)
    .data_parallel(n_process=2)
    .enable_autocast_fp16()
    .expected_eval_acc(approx(0.5, abs=0.5))
    .expected_rela_sparsity(approx(0.5, abs=0.5)),

    "mrpc_cuda_2proc_ddp": deepcopy(mrpc_movement_desc_template)
    .batch_size(64)
    .distributed_data_parallel(n_process=2)
    .expected_eval_acc(approx(0.5, abs=0.5))
    .expected_rela_sparsity(approx(0.5, abs=0.5)),

    "mrpc_cuda_2proc_ddp_fp16": deepcopy(mrpc_movement_desc_template)
    .batch_size(64)
    .distributed_data_parallel(n_process=2)
    .enable_autocast_fp16()
    .expected_eval_acc(approx(0.5, abs=0.5))
    .expected_rela_sparsity(approx(0.5, abs=0.5)),

    "mrpc_cpu_1proc": deepcopy(mrpc_movement_desc_template)
    .cpu_only()
    .expected_eval_acc(approx(0.5, abs=0.5))
    .expected_rela_sparsity(approx(0.5, abs=0.5)),
}


def finalize_desc(desc, is_long_training, dataset_dir, tmp_path_factory, weekly_models_path):
    if is_long_training and (weekly_models_path is None):
        pytest.skip('Skip the test for long training since `--weekly-models` option is not specified.')
    if (not is_long_training) and (weekly_models_path is not None):
        pytest.skip('Skip the test for short training since a long run will be checked.')
    return desc.finalize(dataset_dir, tmp_path_factory, weekly_models_path)


@pytest.fixture(
    name="movement_desc_long", scope="module", params=MOVEMENT_DESCRIPTORS.values(), ids=list(MOVEMENT_DESCRIPTORS.keys())
)
def fixture_movement_desc_long(request, dataset_dir, tmp_path_factory, weekly_models_path):
    desc: MovementTrainingTestDescriptor = request.param
    return finalize_desc(desc, True, dataset_dir, tmp_path_factory, weekly_models_path)


@pytest.fixture(
    name="movement_desc_short", scope="module", params=MOVEMENT_DESCRIPTORS.values(), ids=list(MOVEMENT_DESCRIPTORS.keys())
)
def fixture_movement_desc_short(request, dataset_dir, tmp_path_factory, weekly_models_path):
    desc: MovementTrainingTestDescriptor = request.param
    desc = deepcopy(desc).quick_check()
    return finalize_desc(desc, False, dataset_dir, tmp_path_factory, weekly_models_path)


class TestMovementTraining:
    def test_compression_movement_long_train(self, movement_desc_long: MovementTrainingTestDescriptor, mocker):
        if (not movement_desc_long.cpu_only_) and torch.cuda.device_count() < movement_desc_long.n_process:
            pytest.skip(f"No enough cuda devices to run {movement_desc_long}")
        validator = movement_desc_long.get_validator()
        args = validator.get_default_args()
        validator.validate_sample(args, mocker)
        self._validate_model_is_saved(movement_desc_long)
        self._validate_train_metric(movement_desc_long)

    def test_compression_movement_short_train(self, movement_desc_short: MovementTrainingTestDescriptor, mocker):
        if (not movement_desc_short.cpu_only_) and torch.cuda.device_count() < movement_desc_short.n_process:
            pytest.skip(f"No enough cuda devices to run {movement_desc_short}")
        validator = movement_desc_short.get_validator()
        args = validator.get_default_args()
        validator.validate_sample(args, mocker)
        self._validate_model_is_saved(movement_desc_short)

    @staticmethod
    def _validate_model_is_saved(desc: MovementTrainingTestDescriptor):
        assert Path(desc.output_dir, 'pytorch_model.bin').is_file()

    @staticmethod
    def _validate_train_metric(desc: MovementTrainingTestDescriptor):
        metrics = desc.get_metric()
        if desc.expected_eval_acc_ is not None:
            assert metrics["eval_accuracy"] == approx(desc.expected_eval_acc_)
        if desc.expected_eval_f1_ is not None:
            assert metrics["eval_f1"] == approx(desc.expected_eval_f1_)
        if desc.expected_rela_sparsity_ is not None:
            assert metrics["movement_sparsity/relative_sparsity"] == approx(
                desc.expected_rela_sparsity_)
