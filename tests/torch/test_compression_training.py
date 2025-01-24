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

import json
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest
from pytest import approx

from nncf import NNCFConfig
from tests.cross_fw.shared.helpers import get_cli_dict_args
from tests.cross_fw.shared.paths import PROJECT_ROOT
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.helpers import Command
from tests.torch.sample_test_validator import BaseSampleTestCaseDescriptor
from tests.torch.sample_test_validator import BaseSampleValidator
from tests.torch.sample_test_validator import SampleType
from tests.torch.test_sanity_sample import update_compression_algo_dict_with_legr_save_load_params


class CompressionTrainingValidator(BaseSampleValidator):
    def __init__(self, desc: "CompressionTrainingTestDescriptor"):
        self._desc = desc
        self._sample_handler = desc.sample_handler

    def setup_spy(self, mocker):
        pass

    def validate_spy(self):
        pass

    def validate_sample(self, args, mocker):
        cli_args = get_cli_dict_args(args)
        cmd = self._create_command_line(cli_args)
        env_with_cuda_reproducibility = os.environ.copy()
        env_with_cuda_reproducibility["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env_with_cuda_reproducibility["PYTHONPATH"] = str(PROJECT_ROOT)
        runner = Command(cmd, env=env_with_cuda_reproducibility)
        runner.run(timeout=self._desc.timeout_)

    def get_default_args(self, tmp_path):
        args = {
            "mode": "train",
            "data": str(self._desc.dataset_dir),
            "config": str(self._desc.config_path),
            "log-dir": str(tmp_path),
            "workers": 0,  # Workaround for PyTorch MultiprocessingDataLoader issues
        }
        if self._desc.seed is not None:
            args["seed"] = self._desc.seed
        if self._desc.weights_filename_ is not None:
            args["weights"] = self._desc.weights_path
        if self._desc.checkpoint_save_dir is not None:
            args["checkpoint-save-dir"] = self._desc.checkpoint_save_dir
        if self._desc.execution_arg:
            args[self._desc.execution_arg] = None
        if self._desc.mixed_precision:
            args["mixed-precision"] = None
        return args

    def _create_command_line(self, args):
        executable = self._sample_handler.get_executable()
        cli_args = " ".join(
            key if (val is None or val is True) else "{} {}".format(key, val) for key, val in args.items()
        )
        return f"{sys.executable} {executable} {cli_args}"


class CompressionTrainingTestDescriptor(BaseSampleTestCaseDescriptor):
    def __init__(self):
        super().__init__()
        self.sample_type(SampleType.CLASSIFICATION)
        self.real_dataset("cifar100")
        self.execution_arg = None
        self.distributed_data_parallel()
        self.expected_accuracy_ = None
        self.absolute_tolerance_train_ = 1.0
        self.absolute_tolerance_eval_ = 2e-2
        self.better_accuracy_tolerance = 3
        self.checkpoint_save_dir = None
        self.checkpoint_name = None
        self.seed = 1
        self._mixed_precision = False
        self.weights_filename_ = None
        self.weights_path = None
        self.timeout_ = 30 * 60  # 30 min

    def __str__(self):
        config_name = self.config_name_.replace(".json", "")
        execution_arg = self.execution_arg
        if not execution_arg:
            execution_arg = "data_parallel"
        return "_".join([config_name, self.dataset_name, execution_arg])

    @property
    def config_directory(self) -> Path:
        sample_dir_name = self.sample_handler.get_sample_dir_name()
        return TEST_ROOT / "torch" / "data" / "configs" / "weekly" / sample_dir_name / self.dataset_name

    def get_main_filename(self) -> str:
        return self.sample_handler.get_main_location().split(".")[-1] + ".py"

    def get_checkpoint_path(self) -> str:
        return self.sample_handler.get_checkpoint_path(self.checkpoint_save_dir, self.checkpoint_name, self.config_path)

    def get_validator(self):
        return CompressionTrainingValidator(self)

    def expected_accuracy(self, expected_accuracy: float):
        self.expected_accuracy_ = expected_accuracy
        return self

    def weights_filename(self, filename: str):
        self.weights_filename_ = filename
        return self

    def absolute_tolerance_train(self, tolerance: float):
        self.absolute_tolerance_train_ = tolerance
        return self

    def absolute_tolerance_eval(self, tolerance: float):
        self.absolute_tolerance_eval_ = tolerance
        return self

    def distributed_data_parallel(self):
        self.execution_arg = "multiprocessing-distributed"
        return self

    def cpu_only(self):
        self.execution_arg = "cpu-only"
        return self

    @property
    def mixed_precision(self):
        return self._mixed_precision

    def use_mixed_precision(self):
        self._mixed_precision = True

    def data_parallel(self):
        self.execution_arg = ""
        return self

    def no_seed(self):
        self.seed = None
        return self

    def timeout_seconds(self, num_seconds: int):
        self.timeout_ = num_seconds
        return self

    def finalize(self, dataset_dir, tmp_path_factory, weekly_models_path) -> "CompressionTrainingTestDescriptor":
        if self.dataset_dir is None:
            self.dataset_dir = Path(
                dataset_dir if dataset_dir else os.path.join(tempfile.gettempdir(), self.dataset_name)
            )
        self.weights_path = self._get_weight_path(weekly_models_path)
        if self.weights_path is not None:
            assert os.path.exists(self.weights_path), "Weights file does not exist: {}".format(self.weights_path)
        checkpoint_save_dir = str(tmp_path_factory.mktemp("models"))
        self.checkpoint_save_dir = os.path.join(checkpoint_save_dir, self.execution_arg.replace("-", "_"))
        return self

    def get_metric(self):
        return self.sample_handler.get_metric_value_from_checkpoint(
            self.checkpoint_save_dir, self.checkpoint_name, self.config_path
        )

    def _get_weight_path(self, weekly_models_path):
        if self.weights_filename_ is None:
            return None
        return os.path.join(weekly_models_path, self.sample_type_.value, self.dataset_name, self.weights_filename_)


class LEGRTrainingTestDescriptor(CompressionTrainingTestDescriptor):
    def __init__(self):
        super().__init__()
        self.num_train_steps = 0
        self._config_path = None
        self.timeout_ = 3 * 60 * 60  # 3 hours, because legr training takes 2.5-3 hours

    @property
    def config_path(self) -> Path:
        return self._config_path

    def update_compression_config_with_legr_save_load_params(self, tmp_path, save=True):
        nncf_config_path = str(super().config_path)
        nncf_config = NNCFConfig.from_json(nncf_config_path)
        updated_nncf_config = update_compression_algo_dict_with_legr_save_load_params(
            deepcopy(nncf_config), tmp_path, save
        )
        new_nncf_config_path = nncf_config_path
        if updated_nncf_config != nncf_config:
            new_nncf_config_path = os.path.join(tmp_path, os.path.basename(nncf_config_path))
            with open(str(new_nncf_config_path), "w", encoding="utf8") as f:
                json.dump(updated_nncf_config, f)
        self._config_path = Path(new_nncf_config_path)
        return new_nncf_config_path


class NASTrainingValidator(CompressionTrainingValidator):
    def __init__(self, desc: "NASTrainingTestDescriptor"):
        super().__init__(desc)
        self._desc = desc  # to override typehint to a child class

    def get_default_args(self, tmp_path):
        args = super().get_default_args(tmp_path)
        if self._desc.num_train_steps_:
            args["train-steps"] = self._desc.num_train_steps_
        return args

    def validate_subnet(self):
        ref_acc = self._desc.subnet_expected_accuracy_
        act_acc = self._desc.get_subnet_metric()
        assert act_acc > ref_acc


class NASTrainingTestDescriptor(CompressionTrainingTestDescriptor):
    def __init__(self):
        super().__init__()
        self.sample_type(SampleType.CLASSIFICATION_NAS)
        self.data_parallel()
        self.num_train_steps_ = None
        self.checkpoint_name = "supernet"
        self.subnet_expected_accuracy_ = None
        self.subnet_checkpoint_name = "subnetwork"

    def num_train_steps(self, num_steps: int):
        self.num_train_steps_ = num_steps
        return self

    def get_validator(self) -> "NASTrainingValidator":
        return NASTrainingValidator(self)

    def subnet_expected_accuracy(self, subnet_expected_accuracy: float):
        self.subnet_expected_accuracy_ = subnet_expected_accuracy
        return self

    def get_subnet_metric(self):
        return self.sample_handler.get_metric_value_from_checkpoint(
            self.checkpoint_save_dir, self.subnet_checkpoint_name
        )

    def _get_weight_path(self, weekly_models_path):
        return os.path.join(
            weekly_models_path, SampleType.CLASSIFICATION.value, self.dataset_name, self.weights_filename_
        )


# No seed to workaround PyTorch 1.9.1 multiprocessing issue related to determinism and asym quantization.
# https://github.com/pytorch/pytorch/issues/61032
MOBILENET_V2_ASYM_INT8 = (
    CompressionTrainingTestDescriptor()
    .config_name("mobilenet_v2_asym_int8.json")
    .expected_accuracy(68.11)
    .weights_filename("mobilenet_v2_32x32_cifar100_68.11.pth")
    .absolute_tolerance_train(1.0)
    .absolute_tolerance_eval(5e-1)  # return to 2e-1 after cu121 migration. Ticket 124083.
    .no_seed()
)

MOBILENET_V2_MAGNITUDE_SPARSITY_INT8 = (
    CompressionTrainingTestDescriptor()
    .config_name("mobilenet_v2_magnitude_sparsity_int8.json")
    .expected_accuracy(68.11)
    .weights_filename("mobilenet_v2_32x32_cifar100_68.11.pth")
    .absolute_tolerance_train(1.5)
    .absolute_tolerance_eval(5e-1)  # return to 2e-1 after cu121 migration. Ticket 124083.
)

QUANTIZATION_DESCRIPTORS = [
    CompressionTrainingTestDescriptor()
    .config_name("mobilenet_v2_sym_int8.json")
    .expected_accuracy(68.11)
    .weights_filename("mobilenet_v2_32x32_cifar100_68.11.pth")
    .absolute_tolerance_train(1.0)
    .absolute_tolerance_eval(5e-1),  # return to 2e-1 after cu121 migration. Ticket 124083.
    MOBILENET_V2_ASYM_INT8,
    deepcopy(MOBILENET_V2_ASYM_INT8).cpu_only(),
    CompressionTrainingTestDescriptor()
    .config_name("inceptionV3_int8.json")
    .expected_accuracy(77.53)
    .weights_filename("inceptionV3_77.53.sd")
    .data_parallel()
    .absolute_tolerance_eval(2e-1)
    .timeout_seconds(60 * 60),  # 1 hour
    CompressionTrainingTestDescriptor()
    .config_name("resnet50_int8.json")
    .expected_accuracy(67.93)
    .data_parallel()
    .weights_filename("resnet50_cifar100_67.93.pth")
    .absolute_tolerance_eval(2e-1),
]

SPARSITY_DESCRIPTORS = [
    MOBILENET_V2_MAGNITUDE_SPARSITY_INT8,
    deepcopy(MOBILENET_V2_MAGNITUDE_SPARSITY_INT8).data_parallel().timeout_seconds(60 * 60),  # 1 hour,
    CompressionTrainingTestDescriptor()
    .config_name("mobilenet_v2_rb_sparsity_int8.json")
    .expected_accuracy(68.11)
    .weights_filename("mobilenet_v2_32x32_cifar100_68.11.pth")
    .absolute_tolerance_eval(5e-1)  # return to 1.5e-1 after cu121 migration. Ticket 124083.
    .timeout_seconds(2 * 60 * 60),  # 2 hours
]

NAS_DESCRIPTORS = [
    NASTrainingTestDescriptor()
    .real_dataset("cifar10")
    .config_name("mobilenet_v2_nas_SMALL.json")
    .expected_accuracy(85.1)
    .subnet_expected_accuracy(88.67)
    .weights_filename("mobilenet_v2_cifar10_93.91.pth")
    .absolute_tolerance_train(1.0)
    .absolute_tolerance_eval(5e-1),  # return to 2e-2 after cu121 migration. Ticket 124083.
    NASTrainingTestDescriptor()
    .real_dataset("cifar10")
    .config_name("resnet50_nas_SMALL.json")
    .subnet_expected_accuracy(88.67)
    .expected_accuracy(85.19)
    .weights_filename("resnet50_cifar10_93.65.pth")
    .absolute_tolerance_train(2.0)
    .absolute_tolerance_eval(2e-2),
    NASTrainingTestDescriptor()
    .real_dataset("cifar10")
    .config_name("vgg11_bn_nas_SMALL.json")
    .subnet_expected_accuracy(85.09)
    .expected_accuracy(89.43)
    .weights_filename("vgg11_bn_cifar10_92.39.pth")
    .absolute_tolerance_train(2.0)
    .absolute_tolerance_eval(2e-2),
    # NASTrainingTestDescriptor()
    #     .real_dataset('cifar100')
    #     .config_name('efficient_net_b0_nas_SMALL.json')
    #     .expected_accuracy(1)
    #     .subnet_expected_accuracy(88.67)
    #     .weights_filename('efficient_net_b0_cifar100_87.02.pth')
    #     .absolute_tolerance_train(1.0)
    #     .absolute_tolerance_eval(2e-2)
]

IMAGENET_DESCRIPTORS = [
    CompressionTrainingTestDescriptor()
    .config_name("mobilenet_v2_sym_int8.json")
    .real_dataset("imagenet")
    .expected_accuracy(100)
    .data_parallel()
    .weights_filename("mobilenet_v2.pth.tar"),
    CompressionTrainingTestDescriptor()
    .config_name("mobilenet_v2_sym_int8.json")
    .real_dataset("imagenet")
    .expected_accuracy(100)
    .weights_filename("mobilenet_v2.pth.tar"),
    CompressionTrainingTestDescriptor()
    .config_name("mobilenet_v2_asym_int8.json")
    .real_dataset("imagenet")
    .expected_accuracy(100)
    .weights_filename("mobilenet_v2.pth.tar"),
    CompressionTrainingTestDescriptor()
    .config_name("resnet50_sym_int8.json")
    .real_dataset("imagenet")
    .expected_accuracy(100),
    CompressionTrainingTestDescriptor()
    .config_name("resnet50_asym_int8.json")
    .real_dataset("imagenet")
    .expected_accuracy(100),
]

TEST_CASE_DESCRIPTORS = [
    *QUANTIZATION_DESCRIPTORS,
    *SPARSITY_DESCRIPTORS,
    *IMAGENET_DESCRIPTORS,
]

LEGR_TEST_CASE_DESCRIPTORS = [
    LEGRTrainingTestDescriptor()
    .config_name("mobilenet_v2_learned_ranking.json")
    .expected_accuracy(68.11)
    .weights_filename("mobilenet_v2_32x32_cifar100_68.11.pth")
    .absolute_tolerance_train(1.5)
    .absolute_tolerance_eval(5e-1),  # return to 3e-2 after cu121 migration. Ticket 124083.
]


@pytest.fixture(name="case_common_dirs", scope="module")
def fixture_case_common_dirs(tmp_path_factory):
    return {
        "save_coeffs_path": str(tmp_path_factory.mktemp("ranking_coeffs")),
    }


def finalize_desc(desc, dataset_dir, tmp_path_factory, weekly_models_path, enable_imagenet):
    if weekly_models_path is None:
        pytest.skip("Path to models weights for weekly testing is not set, use --weekly-models option.")
    if "imagenet" in desc.dataset_name and not enable_imagenet:
        pytest.skip("ImageNet tests were intentionally skipped as it takes a lot of time")
    return desc.finalize(dataset_dir, tmp_path_factory, weekly_models_path)


@pytest.fixture(name="desc", scope="module", params=TEST_CASE_DESCRIPTORS, ids=map(str, TEST_CASE_DESCRIPTORS))
def fixture_desc(request, dataset_dir, tmp_path_factory, weekly_models_path, enable_imagenet, mixed_precision):
    desc: CompressionTrainingTestDescriptor = request.param
    if mixed_precision:
        desc.use_mixed_precision()
    return finalize_desc(desc, dataset_dir, tmp_path_factory, weekly_models_path, enable_imagenet)


@pytest.fixture(
    name="legr_desc", scope="module", params=LEGR_TEST_CASE_DESCRIPTORS, ids=map(str, LEGR_TEST_CASE_DESCRIPTORS)
)
def fixture_legr_desc(request, dataset_dir, tmp_path_factory, weekly_models_path, enable_imagenet):
    desc: LEGRTrainingTestDescriptor = request.param
    return finalize_desc(desc, dataset_dir, tmp_path_factory, weekly_models_path, enable_imagenet)


@pytest.fixture(name="nas_desc", scope="module", params=NAS_DESCRIPTORS, ids=map(str, NAS_DESCRIPTORS))
def fixture_nas_desc(request, dataset_dir, tmp_path_factory, weekly_models_path, enable_imagenet):
    desc: NASTrainingTestDescriptor = request.param
    return finalize_desc(desc, dataset_dir, tmp_path_factory, weekly_models_path, enable_imagenet)


@pytest.mark.weekly
class TestCompression:
    @pytest.mark.dependency(name="train")
    def test_compression_train(self, desc: CompressionTrainingTestDescriptor, tmp_path, mocker):
        validator = desc.get_validator()
        args = validator.get_default_args(tmp_path)

        validator.validate_sample(args, mocker)

        self._validate_train_metric(desc)

    @pytest.mark.dependency(depends=["train"])
    def test_compression_eval(self, desc: CompressionTrainingTestDescriptor, tmp_path, mocker):
        validator = desc.get_validator()
        args = validator.get_default_args(tmp_path)
        metric_file_path = self._add_args_for_eval(args, desc, tmp_path)

        validator.validate_sample(args, mocker)

        self._validate_eval_metric(desc, metric_file_path)

    @pytest.mark.dependency(name="legr_train")
    def test_compression_legr_train(self, legr_desc: LEGRTrainingTestDescriptor, tmp_path, mocker, case_common_dirs):
        validator = legr_desc.get_validator()
        args = validator.get_default_args(tmp_path)
        args["config"] = legr_desc.update_compression_config_with_legr_save_load_params(
            case_common_dirs["save_coeffs_path"], True
        )

        validator.validate_sample(args, mocker)

        self._validate_train_metric(legr_desc)

    @pytest.mark.dependency(depends=["legr_train"])
    def test_compression_legr_eval(self, legr_desc: LEGRTrainingTestDescriptor, tmp_path, mocker, case_common_dirs):
        validator = legr_desc.get_validator()
        args = validator.get_default_args(tmp_path)
        metric_file_path = self._add_args_for_eval(args, legr_desc, tmp_path)
        args["config"] = legr_desc.update_compression_config_with_legr_save_load_params(
            case_common_dirs["save_coeffs_path"], False
        )

        validator.validate_sample(args, mocker)

        self._validate_eval_metric(legr_desc, metric_file_path)

    @pytest.mark.dependency(name="nas_train")
    def test_compression_nas_train(self, nas_desc: NASTrainingTestDescriptor, tmp_path, mocker):
        validator = nas_desc.get_validator()
        args = validator.get_default_args(tmp_path)

        validator.validate_sample(args, mocker)

        self._validate_train_metric(nas_desc)
        validator.validate_subnet()

    @pytest.mark.dependency(depends=["nas_train"])
    def test_compression_nas_eval(self, nas_desc: NASTrainingTestDescriptor, tmp_path, mocker):
        validator = nas_desc.get_validator()
        args = validator.get_default_args(tmp_path)
        metric_file_path = self._add_args_for_eval(args, nas_desc, tmp_path)

        validator.validate_sample(args, mocker)

        self._validate_eval_metric(nas_desc, metric_file_path)

    @staticmethod
    def _validate_eval_metric(desc: CompressionTrainingTestDescriptor, metric_file_path):
        with open(str(metric_file_path), encoding="utf8") as metric_file:
            metrics = json.load(metric_file)
            ref_metric = metrics["Accuracy"]
            assert desc.get_metric() == approx(ref_metric, abs=desc.absolute_tolerance_eval_)

    @staticmethod
    def _add_args_for_eval(args, desc, tmp_path):
        args["mode"] = "test"
        checkpoint_path = desc.get_checkpoint_path()
        args["resume"] = checkpoint_path
        if "weights" in args:
            del args["weights"]
        metric_file_path = tmp_path / "metrics.json"
        args["metrics-dump"] = tmp_path / metric_file_path
        return metric_file_path

    @staticmethod
    def _validate_train_metric(desc: CompressionTrainingTestDescriptor):
        ref_acc = desc.expected_accuracy_
        actual_acc = desc.get_metric()
        tolerance = desc.absolute_tolerance_train_ if actual_acc < ref_acc else desc.better_accuracy_tolerance
        assert actual_acc == approx(ref_acc, abs=tolerance)
