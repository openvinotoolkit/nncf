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

import datetime
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import pytest
from pytest import FixtureRequest

import nncf
from tests.cross_fw.shared.metric_thresholds import DIFF_FP32_MAX_GLOBAL
from tests.cross_fw.shared.metric_thresholds import DIFF_FP32_MIN_GLOBAL
from tests.cross_fw.shared.paths import DATASET_DEFINITIONS_PATH
from tests.cross_fw.shared.paths import PROJECT_ROOT
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.helpers import Command

DIFF_TARGET_PT_MIN = -0.1
DIFF_TARGET_PT_MAX = 0.1
DIFF_TARGET_OV_MIN = -0.01
DIFF_TARGET_OV_MAX = 0.01
PYTORCH = "PT"
OPENVINO = "OV"
TRAIN = "TRAIN"


@dataclass
class EvalRunParamsStruct:
    """
    Contain data about quantization of the model.
    """

    config_name: str
    reference: Optional[str]
    target_ov: float
    target_pt: float
    metric_type: str
    dataset_name: str
    sample_type: str
    resume_file: str
    batch: int
    diff_fp32_min: float
    diff_fp32_max: float
    model_name: str
    diff_target_ov_min: float
    diff_target_ov_max: float
    diff_target_pt_min: float
    diff_target_pt_max: float
    multiprocessing_distributed: bool
    skip_ov: Optional[str]
    xfail_ov: Optional[str]


@dataclass
class ResultInfo:
    """
    Contain data about result of test.
    """

    model_name: str
    backend: str
    metric_type: Optional[str] = None
    measured: Optional[float] = None
    expected: Optional[float] = None
    diff_fp32: Optional[float] = None
    target_fp32: Optional[float] = None
    diff_target: Optional[float] = None
    status: Optional[str] = None

    def to_dict(self):
        return {
            "Model": self.model_name,
            "Backend": self.backend,
            "Metrics type": self.metric_type,
            "Measured": self.measured,
            "Expected": self.expected,
            "Diff expected": self.diff_target,
            "Target FP32": self.target_fp32,
            "Diff FP32": self.diff_fp32,
            "Status": self.status,
            "Build url": os.environ.get("BUILD_URL", ""),
        }


def read_reference_file(ref_path: Path) -> List[EvalRunParamsStruct]:
    """
    Reads the reference file to get a list of `EvalRunParamsStruct` objects.

    :param ref_path: The path to the JSON reference file.
    :return: A list of `EvalRunParamsStruct` objects.
    """

    with ref_path.open(encoding="UTF-8") as source:
        sota_eval_config = json.load(source, object_pairs_hook=OrderedDict)

    param_list = []
    model_names = []
    for sample_type_ in sota_eval_config:
        datasets = sota_eval_config[sample_type_]
        for dataset_name in datasets:
            model_dict = datasets[dataset_name]
            for model_name, sample_dict in model_dict.items():
                if model_name in model_names:
                    raise nncf.InternalError(f"Model name {model_name} is not unique.")
                model_names.append(model_name)
                param_list.append(
                    EvalRunParamsStruct(
                        model_name=model_name,
                        config_name=sample_dict["config"],
                        reference=sample_dict.get("reference", None),
                        target_pt=sample_dict["target_pt"],
                        target_ov=sample_dict["target_ov"],
                        metric_type=sample_dict["metric_type"],
                        dataset_name=dataset_name,
                        sample_type=sample_type_,
                        resume_file=sample_dict.get("resume", None),
                        batch=sample_dict.get("batch", None),
                        diff_fp32_min=sample_dict.get("diff_fp32_min", DIFF_FP32_MIN_GLOBAL),
                        diff_fp32_max=sample_dict.get("diff_fp32_max", DIFF_FP32_MAX_GLOBAL),
                        diff_target_ov_min=sample_dict.get("diff_target_ov_min", DIFF_TARGET_OV_MIN),
                        diff_target_ov_max=sample_dict.get("diff_target_ov_max", DIFF_TARGET_OV_MAX),
                        diff_target_pt_min=sample_dict.get("diff_target_pt_min", DIFF_TARGET_PT_MIN),
                        diff_target_pt_max=sample_dict.get("diff_target_pt_max", DIFF_TARGET_PT_MAX),
                        multiprocessing_distributed=sample_dict.get("multiprocessing_distributed", False),
                        skip_ov=sample_dict.get("skip_ov", None),
                        xfail_ov=sample_dict.get("xfail_ov", None),
                    )
                )
    return param_list


EVAL_TEST_STRUCT = read_reference_file(TEST_ROOT / "torch" / "sota_checkpoints_eval.json")
REF_PT_FP32_METRIC = {p.model_name: p.target_pt for p in EVAL_TEST_STRUCT if p.reference is None}
REF_OV_FP32_METRIC = {p.model_name: p.target_ov for p in EVAL_TEST_STRUCT if p.reference is None}


def idfn(val):
    if isinstance(val, EvalRunParamsStruct):
        return val.model_name


def generate_run_examples_command(
    sample_type: str,
    mode: str,
    config: str,
    dataset_path: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    metrics_dump_file_path: Optional[Path] = None,
    multiprocessing_distributed: bool = False,
    resume_file_path: Optional[Path] = None,
    weights_path: Optional[Path] = None,
    export_model_path: Optional[Path] = None,
    batch: Optional[int] = None,
    cpu_only: bool = False,
    checkpoint_dir: Optional[Path] = None,
    distributed_mode_sync_port: Optional[str] = None,
) -> str:
    """
    Generates a command line to run script `tests/torch/run_examples_for_test_sota.py`.

    :param sample_type: The type of sample to run.
    :param mode: The mode to run the example in (e.g., train, eval, export).
    :param config: The path to the configuration file.
    :param dataset_path: The path to the dataset directory.
    :param log_dir: The path to the log directory.
    :param metrics_dump_file_path: The path to the metrics dump file.
    :param multiprocessing_distributed: Whether to use multiprocessing distributed training.
    :param resume_file_path: The path to the resume file.
    :param weights_path: The path to the weights file.
    :param export_model_path: The path to the export model directory.
    :param batch: The batch size.
    :param cpu_only: Whether to use the CPU only.
    :param checkpoint_dir: The path to the checkpoint directory.
    :param distributed_mode_sync_port: The port to use for distributed mode synchronization.
    :return: A command line to run the run_examples_for_test_sota.py script.
    """
    cmd = [
            sys.executable,
            "tests/torch/run_examples_for_test_sota.py",
            sample_type,
            "-m", mode,
            "--config", config,
        ]  # fmt: skip
    if dataset_path is not None:
        cmd += ["--data", dataset_path.as_posix()]
    if resume_file_path is not None:
        cmd += ["--resume", resume_file_path.as_posix()]
    else:
        cmd += ["--pretrained"]
    if weights_path is not None and weights_path.exists():
        cmd += ["--weights", weights_path.as_posix()]
    if export_model_path is not None:
        cmd += ["--export-model-path", export_model_path.as_posix()]
    if metrics_dump_file_path is not None:
        cmd += ["--metrics-dump", metrics_dump_file_path.as_posix()]
    if log_dir is not None:
        cmd += ["--log-dir", log_dir.as_posix()]
    if batch is not None:
        cmd += ["-b", str(batch)]
    if multiprocessing_distributed:
        cmd += ["--multiprocessing-distributed"]
    if cpu_only:
        cmd += ["--cpu-only"]
    if checkpoint_dir:
        cmd += ["--checkpoint-save-dir", checkpoint_dir.as_posix()]
    if distributed_mode_sync_port is not None:
        print(f"Setting distributed mode synchronization URL to tcp://127.0.0.1:{distributed_mode_sync_port}")
        cmd += [f"--dist-url=tcp://127.0.0.1:{distributed_mode_sync_port}"]
    return " ".join(cmd)


@pytest.fixture(scope="module")
def metrics_dump_dir(request: FixtureRequest):
    """
    Path to collect metrics from the tests.
    To set this by pytest argument use '--metrics-dump-path'.
    By default metrics_dump_dir is `PROJECT_ROOT/test_results/metrics_dump_YYYY_MM_DD_HH_MM_SS`.
    """
    dump_path = request.config.getoption("--metrics-dump-path")

    if dump_path is None:
        data = datetime.datetime.now()
        dump_path = (
            PROJECT_ROOT / "test_results" / "metrics_dump_"
            f"{'_'.join([str(getattr(data, atr)) for atr in ['year', 'month', 'day', 'hour', 'minute', 'second']])}"
        )

    dump_path.mkdir(exist_ok=True, parents=True)
    assert not dump_path.is_dir() or not next(
        dump_path.iterdir(), None
    ), f"metrics_dump_path dir should be empty: {dump_path}"
    print(f"metrics_dump_path: {dump_path}")
    return dump_path


@pytest.mark.nightly
class TestSotaCheckpoints:
    """
    Test examples for PyTorch compression and checkpoints that provided
    in https://github.com/openvinotoolkit/nncf/blob/develop/docs/ModelZoo.md#pytorch
    """

    @pytest.fixture(scope="class")
    def collected_data(self, metrics_dump_dir):
        """
        Fixture to collect information about tests in `ResultInfo` struct
        and dump it to `metrics_dump_dir / results.csv`.
        """
        data: List[ResultInfo] = []
        yield data
        if metrics_dump_dir and data:
            path = metrics_dump_dir / "results.csv"
            data_frame = pd.DataFrame.from_records([x.to_dict() for x in data])
            data_frame = data_frame.sort_values("Model").reset_index(drop=True)
            data_frame.to_csv(path, index=False)
            print(f"Result file: {path}")

    @pytest.fixture(params=EVAL_TEST_STRUCT, ids=idfn)
    def eval_run_param(self, request: FixtureRequest) -> EvalRunParamsStruct:
        """
        Returns the test cases that were built from the `tests/torch/sota_checkpoints_eval.json` file.
        """
        return request.param

    @staticmethod
    def get_metric_file_name(metrics_dump_path: Path, model_name: str) -> Path:
        """
        Returns the path to the file that contains the metrics for the target model.

        :param metrics_dump_path: The directory that contains the metrics from the test evaluation.
        :param model_name: The name of the target model.
        :return: The path to the file that contains the metrics for the target model.
        """
        return metrics_dump_path / f"{model_name}.metrics.json"

    @staticmethod
    def read_metric(metric_file_name: str) -> float:
        """
        Reads the metric value from the given metric file.

        :param metric_file_name: Path to the metric file.
        :return: The metric value.
        """
        with open(metric_file_name, encoding="utf8") as metric_file:
            metrics = json.load(metric_file)
        return metrics["Accuracy"]

    @staticmethod
    def generate_accuracy_check_cmd(
        config_path: Path, ov_data_dir: Path, model_folder: Path, report_csv_path: Path
    ) -> str:
        """
        Generates a command line to run the accuracy_check tool.

        :param config_path: Path to the config file for the accuracy checker.
        :param ov_data_dir: Path to the dataset directory.
        :param model_folder: Directory that contains the target model in OpenVINO format.
        :param report_csv_path: Path to the report file.
        :return: A command line to run the accuracy_check tool.
        """
        cmd = [
            "python -m accuracy_checker.main",
            "--config", config_path.as_posix(),
            "--source", ov_data_dir.as_posix(),
            "--definitions", DATASET_DEFINITIONS_PATH.as_posix(),
            "--models", model_folder.as_posix(),
            "--csv_result", report_csv_path.as_posix(),
        ]  # fmt: skip
        return " ".join(cmd)

    def get_reference_fp32_metric(self, metrics_dump_path: Path, reference_name: str) -> Tuple[Optional[float], bool]:
        """
        Get reference metric to not compressed model.
        In case of exists reference data will get reference metric from it others reference data gets
        from `tests/torch/sota_checkpoints_eval.json`.

        :param metrics_dump_path: Directory that collect in metric data.
        :param reference_name: Name of the target model.
        :return: Reference metric.
        """
        fp32_metric = None
        if reference_name is not None:
            fp32_metric = REF_PT_FP32_METRIC[reference_name]
            reference_metric_file_path = self.get_metric_file_name(metrics_dump_path, reference_name)
            if reference_metric_file_path.exists():
                acc = self.read_metric(reference_metric_file_path)
                if acc:
                    fp32_metric = acc

        return fp32_metric

    @staticmethod
    def threshold_check(
        diff_target: float,
        diff_target_min: float,
        diff_target_max: float,
        diff_fp32: Optional[float] = None,
        diff_fp32_min: Optional[float] = None,
        diff_fp32_max: Optional[float] = None,
    ) -> Optional[str]:
        """
        Checks whether the difference meets the target thresholds.
        If the difference is not within the target thresholds, the method returns an error message.
        Otherwise, the method returns `None`.
        """
        err_msgs = []
        if diff_target < diff_target_min or diff_target > diff_target_max:
            err_msgs.append(
                "Target diff is not within thresholds: " + f"{diff_target_min} < {diff_target} < {diff_target_max}"
            )
        if diff_fp32 is not None and (diff_fp32 < diff_fp32_min or diff_fp32 > diff_fp32_max):
            err_msgs.append(f"FP32 diff is not within thresholds: {diff_fp32_min} < {diff_fp32} < {diff_fp32_max}")
        if err_msgs:
            return ";".join(err_msgs)
        return None

    @staticmethod
    def get_env():
        """
        Returns a copy of the current environment with the `PYTHONPATH` variable updated
        to include the project root directory
        """
        env = os.environ.copy()
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] += ":" + str(PROJECT_ROOT)
        else:
            env["PYTHONPATH"] = str(PROJECT_ROOT)
        return env

    @pytest.mark.eval
    def test_eval(
        self,
        sota_checkpoints_dir: str,
        sota_data_dir: str,
        eval_run_param: EvalRunParamsStruct,
        collected_data: List[ResultInfo],
        metrics_dump_dir: Path,
    ):
        """
        Runs a test example to validate the target models on the PyTorch backend.
        """
        if sota_data_dir is None:
            pytest.skip("Path to datasets is not set")

        metrics_dump_file_path = self.get_metric_file_name(metrics_dump_dir, model_name=eval_run_param.model_name)
        log_dir = metrics_dump_dir / "logs"

        resume_file_path = None
        if eval_run_param.resume_file:
            assert sota_checkpoints_dir is not None, "sota_checkpoints_dir is not set"
            resume_file_path = Path(sota_checkpoints_dir) / eval_run_param.resume_file
            assert resume_file_path.exists(), f"{resume_file_path} does not exists"

        cmd = generate_run_examples_command(
            sample_type=eval_run_param.sample_type,
            mode="test",
            config=eval_run_param.config_name,
            dataset_path=Path(sota_data_dir) / eval_run_param.dataset_name,
            log_dir=log_dir,
            metrics_dump_file_path=metrics_dump_file_path,
            multiprocessing_distributed=eval_run_param.multiprocessing_distributed,
            resume_file_path=resume_file_path,
            batch=eval_run_param.batch,
        )

        runner = Command(cmd, cwd=PROJECT_ROOT, env=self.get_env())
        exit_code = runner.run(assert_returncode_zero=False)

        is_ok = exit_code == 0 and metrics_dump_file_path.exists()

        metric_value = None
        diff_target = None
        diff_fp32 = None
        if not is_ok:
            status = f"exit_code: {exit_code}"
            result_info = ResultInfo(
                model_name=eval_run_param.model_name,
                backend=PYTORCH,
                status=status,
            )
            collected_data.append(result_info)
            pytest.fail(status)

        metric_value = self.read_metric(metrics_dump_file_path)
        fp32_metric = self.get_reference_fp32_metric(metrics_dump_dir, eval_run_param.reference)

        diff_target = round((metric_value - eval_run_param.target_pt), 2)
        if fp32_metric:
            diff_fp32 = round((metric_value - fp32_metric), 2)

        threshold_errors = self.threshold_check(
            diff_target=diff_target,
            diff_target_min=eval_run_param.diff_target_pt_min,
            diff_target_max=eval_run_param.diff_target_pt_max,
            diff_fp32=diff_fp32,
            diff_fp32_min=eval_run_param.diff_fp32_min,
            diff_fp32_max=eval_run_param.diff_fp32_max,
        )

        result_info = ResultInfo(
            model_name=eval_run_param.model_name,
            backend=PYTORCH,
            metric_type=eval_run_param.metric_type,
            measured=metric_value,
            expected=eval_run_param.target_pt,
            diff_target=diff_target,
            target_fp32=fp32_metric,
            diff_fp32=diff_fp32,
            status=threshold_errors,
        )
        collected_data.append(result_info)
        if threshold_errors is not None:
            pytest.fail(threshold_errors)

    @staticmethod
    def get_ir_model_path(model_name: str):
        """
        Get path to OpenVINO model by model name.
        """
        return PROJECT_ROOT / "ir_models" / model_name / f"{model_name}.xml"

    @pytest.mark.convert
    def test_convert(self, eval_run_param: EvalRunParamsStruct, openvino: bool, sota_checkpoints_dir: str):
        """
        Runs a test example to convert target models to OpenVINO format.
        """
        if not openvino:
            pytest.skip("Skip if not --run-openvino-eval")
        if eval_run_param.skip_ov:
            pytest.skip("Skipped by 'skip_ov' in sota_checkpoints_eval.json")
        os.chdir(PROJECT_ROOT)
        ir_model_path = self.get_ir_model_path(eval_run_param.model_name)
        resume_file_path = None
        if eval_run_param.resume_file:
            assert sota_checkpoints_dir is not None, "sota_checkpoints_dir is not set"
            resume_file_path = Path(sota_checkpoints_dir) / eval_run_param.resume_file
            assert resume_file_path.exists(), f"{resume_file_path} does not exists"

        cmd = generate_run_examples_command(
            sample_type=eval_run_param.sample_type,
            mode="export",
            config=eval_run_param.config_name,
            cpu_only=True,
            export_model_path=ir_model_path,
            resume_file_path=resume_file_path,
        )
        runner = Command(cmd, cwd=PROJECT_ROOT, env=self.get_env())
        runner.run()

    @staticmethod
    def get_metric_from_ac_csv(path: Path):
        """
        Get metric value from the report of accuracy_checker.

        :param path: Path ot report file of accuracy_checker.
        :return: Metric value.
        """
        data = pd.read_csv(path)
        return round(data["metric_value"].iloc[0] * 100, 2)

    @pytest.mark.oveval
    def test_openvino_eval(
        self,
        eval_run_param: EvalRunParamsStruct,
        ov_data_dir: Path,
        openvino: bool,
        ov_config_dir: str,
        collected_data: List[ResultInfo],
        metrics_dump_dir: Path,
    ):
        """
        Runs a test example to validate the target models on the PyTorch backend.
        """
        if not openvino:
            pytest.skip("Skip if not --run-openvino-eval")
        if ov_data_dir is None:
            pytest.fail("--ov-data-dir is not set")
        if eval_run_param.skip_ov:
            status = f"Skip by: {eval_run_param.skip_ov}"
            collected_data.append(
                ResultInfo(
                    model_name=eval_run_param.model_name,
                    backend=OPENVINO,
                    status=status,
                )
            )
            pytest.skip(status)

        config_folder = ov_config_dir or PROJECT_ROOT / "tests" / "torch" / "data" / "ac_configs"
        ir_model_path = self.get_ir_model_path(eval_run_param.model_name)

        if not ir_model_path.exists():
            collected_data.append(
                ResultInfo(
                    model_name=eval_run_param.model_name,
                    backend=OPENVINO,
                    status="IR does not exists",
                )
            )
            pytest.fail("IR does not exists")

        ac_yml_path = config_folder / f"{eval_run_param.model_name}.yml"
        report_csv_path = metrics_dump_dir / f"{eval_run_param.model_name}.csv"

        # Ensure that report file does not exists
        report_csv_path.unlink(missing_ok=True)

        cmd = self.generate_accuracy_check_cmd(ac_yml_path, ov_data_dir, ir_model_path.parent, report_csv_path)
        runner = Command(cmd, cwd=PROJECT_ROOT, env=self.get_env())
        exit_code = runner.run(assert_returncode_zero=False)

        if exit_code:
            status = f"Accuracy checker return code: {exit_code}"
            collected_data.append(
                ResultInfo(
                    model_name=eval_run_param.model_name,
                    backend=OPENVINO,
                    status=status,
                )
            )
            pytest.fail(status)

        metric_value = self.get_metric_from_ac_csv(report_csv_path)
        fp32_metric = REF_OV_FP32_METRIC.get(eval_run_param.reference, None)

        diff_target = round((metric_value - eval_run_param.target_ov), 2)
        diff_fp32 = None
        if fp32_metric:
            diff_fp32 = round((metric_value - fp32_metric), 2)

        threshold_errors = self.threshold_check(
            diff_target=diff_target,
            diff_target_min=eval_run_param.diff_target_ov_min,
            diff_target_max=eval_run_param.diff_target_ov_max,
            diff_fp32=diff_fp32,
            diff_fp32_min=eval_run_param.diff_fp32_min,
            diff_fp32_max=eval_run_param.diff_fp32_max,
        )
        status = threshold_errors
        if eval_run_param.xfail_ov is not None and threshold_errors is not None:
            status = f"XFAIL: {eval_run_param.xfail_ov} {threshold_errors}"

        result_info = ResultInfo(
            model_name=eval_run_param.model_name,
            backend=OPENVINO,
            metric_type=eval_run_param.metric_type,
            measured=metric_value,
            expected=eval_run_param.target_ov,
            diff_target=diff_target,
            target_fp32=fp32_metric,
            diff_fp32=diff_fp32,
            status=status,
        )

        collected_data.append(result_info)
        if status is not None:
            if eval_run_param.xfail_ov is not None:
                pytest.xfail(status)
            else:
                pytest.fail(status)

    @pytest.mark.train
    def test_train(
        self,
        eval_run_param: EvalRunParamsStruct,
        distributed_mode_sync_port: str,
        sota_data_dir: Path,
        sota_checkpoints_dir: Path,
        collected_data: List[ResultInfo],
        metrics_dump_dir: Path,
    ):
        """
        Runs a test example to train target metric metric is within the 1% threshold.
        """
        if sota_data_dir is None:
            pytest.skip("Path to datasets is not set")

        if eval_run_param.reference is None:
            pytest.skip("Only compressed models must be trained")

        metric_file_name = self.get_metric_file_name(metrics_dump_dir, eval_run_param.model_name)
        metrics_dump_file_path = metrics_dump_dir / metric_file_name
        log_dir = metrics_dump_dir / "logs"
        checkpoint_dir = metrics_dump_dir / "checkpoints"
        weights_path = sota_checkpoints_dir / f"{eval_run_param.reference}.pth"

        cmd = generate_run_examples_command(
            sample_type=eval_run_param.sample_type,
            mode="train",
            config=eval_run_param.config_name,
            dataset_path=sota_data_dir / eval_run_param.dataset_name,
            metrics_dump_file_path=metrics_dump_file_path,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            distributed_mode_sync_port=distributed_mode_sync_port,
            weights_path=weights_path,
        )
        runner = Command(cmd, cwd=PROJECT_ROOT, env=self.get_env())
        exit_code = runner.run(assert_returncode_zero=False)

        is_ok = exit_code == 0 and metrics_dump_file_path.exists()
        err_msg = None
        if is_ok:
            fp32_metric = REF_PT_FP32_METRIC[eval_run_param.reference, None]
            metric_value = self.read_metric(str(metrics_dump_file_path))
            diff_fp32 = round((metric_value - fp32_metric), 2)
            if diff_fp32 > -1:
                err_msg = f"FP32 diff is not within thresholds: -1 < {diff_fp32}"

        collected_data.append(
            ResultInfo(
                model_name=eval_run_param.model_name,
                backend=TRAIN,
                metric_type=eval_run_param.metric_type,
                measured=metric_value,
                target_fp32=fp32_metric,
                diff_fp32=diff_fp32,
                status=err_msg,
            )
        )
        if err_msg:
            pytest.fail(err_msg)
