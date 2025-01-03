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
from typing import Dict, List, Optional, Tuple, Union

import openvino as ov
import pandas as pd
import pytest
import tensorflow as tf
import yaml
from pytest import FixtureRequest

from tests.cross_fw.shared.command import Command
from tests.cross_fw.shared.metric_thresholds import DIFF_FP32_MAX_GLOBAL
from tests.cross_fw.shared.metric_thresholds import DIFF_FP32_MIN_GLOBAL
from tests.cross_fw.shared.openvino_version import get_openvino_version
from tests.cross_fw.shared.paths import DATASET_DEFINITIONS_PATH
from tests.cross_fw.shared.paths import PROJECT_ROOT
from tests.cross_fw.shared.paths import TEST_ROOT

DIFF_TARGET_TF_MIN = -0.1
DIFF_TARGET_TF_MAX = 0.1
DIFF_TARGET_OV_MIN = -0.01
DIFF_TARGET_OV_MAX = 0.01


EVAL_SCRIPT_NAME_MAP = {
    "classification": "main.py",
    "object_detection": "main.py",
    "segmentation": "evaluation.py",
}

PRETRAINED_PARAM_AVAILABILITY = {
    "classification": True,
    "object_detection": False,
    "segmentation": False,
}

DATASET_TYPE_AVAILABILITY = {
    "classification": True,
    "object_detection": True,
    "segmentation": False,
}

num_gpus = len(tf.config.list_physical_devices("GPU"))
BATCH_COEFF = num_gpus if num_gpus else 1


@dataclass
class EvalRunParamsStruct:
    """
    Contain data about quantization of the model.
    """

    model_name: str
    config: Path
    reference: Optional[str]
    target_tf: float
    target_ov: float
    metric_type: str
    dataset_name: str
    dataset_types: List[str]
    sample_type: str
    resume_file: Optional[Path]
    weights: Optional[str]
    batch: Optional[int]
    diff_fp32_min: float
    diff_fp32_max: float
    diff_target_tf_min: float
    diff_target_tf_max: float
    diff_target_ov_min: float
    diff_target_ov_max: float
    skip_ov: Optional[str]
    skip_ov_version: Optional[str]
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
            for model_name, sample_dict in model_dict["topologies"].items():
                if model_name in model_names:
                    raise RuntimeError(f"Model name {model_name} is not unique.")
                model_names.append(model_name)
                batch = sample_dict.get("batch_per_gpu")
                resume = sample_dict.get("resume")
                param_list.append(
                    EvalRunParamsStruct(
                        model_name=model_name,
                        config=Path(sample_dict["config"]),
                        reference=sample_dict.get("reference"),
                        weights=sample_dict.get("weights"),
                        target_tf=sample_dict["target_tf"],
                        target_ov=sample_dict["target_ov"],
                        metric_type=sample_dict["metric_type"],
                        dataset_name=dataset_name,
                        dataset_types=model_dict["dataset_types"],
                        sample_type=sample_type_,
                        resume_file=Path(resume) if resume is not None else None,
                        batch=batch * BATCH_COEFF if batch is not None else None,
                        diff_fp32_min=sample_dict.get("diff_fp32_min", DIFF_FP32_MIN_GLOBAL),
                        diff_fp32_max=sample_dict.get("diff_fp32_max", DIFF_FP32_MAX_GLOBAL),
                        diff_target_ov_min=sample_dict.get("diff_target_ov_min", DIFF_TARGET_OV_MIN),
                        diff_target_ov_max=sample_dict.get("diff_target_ov_max", DIFF_TARGET_OV_MAX),
                        diff_target_tf_min=sample_dict.get("diff_target_tf_min", DIFF_TARGET_TF_MIN),
                        diff_target_tf_max=sample_dict.get("diff_target_tf_max", DIFF_TARGET_TF_MAX),
                        skip_ov=sample_dict.get("skip_ov"),
                        skip_ov_version=sample_dict.get("skip_ov_version"),
                        xfail_ov=sample_dict.get("xfail_ov"),
                    )
                )
    return param_list


EVAL_TEST_STRUCT = read_reference_file(TEST_ROOT / "tensorflow" / "sota_checkpoints_eval.json")
REF_PT_FP32_METRIC = {p.model_name: p.target_tf for p in EVAL_TEST_STRUCT if p.reference is None}
REF_OV_FP32_METRIC = {p.model_name: p.target_ov for p in EVAL_TEST_STRUCT if p.reference is None}


def idfn(val):
    if isinstance(val, EvalRunParamsStruct):
        return val.model_name


def generate_run_examples_command(
    sample_type: str,
    mode: str,
    config: Path,
    dataset_path: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    metrics_dump_file_path: Optional[Path] = None,
    resume_file_path: Optional[Path] = None,
    weights: Optional[Path] = None,
    pretrained: bool = False,
    batch: Optional[int] = None,
    dataset_type: Optional[str] = None,
    to_frozen_graph: Optional[Path] = None,
) -> str:
    """
    Generates a command line to run example.
    """
    cmd = [
            sys.executable,
            f"examples/tensorflow/{sample_type}/{EVAL_SCRIPT_NAME_MAP[sample_type]}",
            "-m", mode,
            "--config", config.as_posix(),
        ]  # fmt: skip

    if dataset_path is not None:
        cmd += ["--data", dataset_path.as_posix()]
    if log_dir is not None:
        cmd += ["--log-dir", log_dir.as_posix()]
    if metrics_dump_file_path is not None:
        cmd += ["--metrics-dump", metrics_dump_file_path.as_posix()]
    if resume_file_path is not None:
        assert resume_file_path.exists(), f"{resume_file_path} does not exist"
        cmd += ["--resume", resume_file_path.as_posix()]
    if weights is not None:
        assert weights.exists(), f"{weights} does not exist"
        cmd += ["--weights", weights.as_posix()]
    if pretrained:
        cmd += ["--pretrained"]
    if dataset_type is not None and DATASET_TYPE_AVAILABILITY[sample_type]:
        cmd += ["--dataset-type", dataset_type]
    if batch:
        cmd += ["-b", str(batch)]
    if to_frozen_graph:
        cmd += ["--to-frozen-graph", to_frozen_graph.as_posix()]
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


class TestSotaCheckpoints:
    @pytest.fixture(scope="class")
    def collected_data(self, metrics_dump_dir: Path):
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

    @staticmethod
    def make_ac_config(config_file, resume_model):
        """
        Generate config for accuracy checker.
        """
        base_config = config_file.parent / f"{resume_model}.yml"
        with open(base_config, encoding="utf8") as f:
            template = yaml.safe_load(f)
        template["models"][0]["name"] = config_file.name.replace(".yml", "")
        with open(config_file, "w", encoding="utf8") as f:
            yaml.dump(template, f, default_flow_style=False)

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
    def get_metric_file_name(metrics_dump_path: Path, model_name: str, dataset_type: Optional[str] = None) -> Path:
        """
        Returns the path to the file that contains the metrics for the target model.
        """
        if dataset_type is not None:
            return metrics_dump_path / f"{model_name}_{dataset_type}.metrics.json"
        return metrics_dump_path / f"{model_name}.metrics.json"

    def get_reference_fp32_metric(
        self, metrics_dump_path: Path, reference_name: str, dataset_type: Optional[str] = None
    ) -> Tuple[Optional[float], bool]:
        """
        Get reference metric to not compressed model.
        In case of exists reference data will get reference metric from it others reference data gets
        from `tests/tensorflow/sota_checkpoints_eval.json`.

        :param metrics_dump_path: Directory that collect in metric data.
        :param reference_name: Name of the target model.
        :return: Reference metric.
        """
        fp32_metric = None
        if reference_name is not None:
            fp32_metric = REF_PT_FP32_METRIC[reference_name]
            reference_metric_file_path = self.get_metric_file_name(metrics_dump_path, reference_name, dataset_type)
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
    def get_weight_params(
        eval_test_struct: EvalRunParamsStruct, sota_checkpoints_dir: Path
    ) -> Dict[str, Union[Path, bool]]:
        if eval_test_struct.resume_file is not None:
            return {"resume_file_path": sota_checkpoints_dir / eval_test_struct.resume_file}
        elif eval_test_struct.weights:
            return {"weights": sota_checkpoints_dir / eval_test_struct.weights}
        elif PRETRAINED_PARAM_AVAILABILITY[eval_test_struct.sample_type]:
            return {"pretrained": True}
        raise RuntimeError("Incorrect config")

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
    @pytest.mark.parametrize("eval_test_struct", EVAL_TEST_STRUCT, ids=idfn)
    @pytest.mark.parametrize("dataset_type", ("tfds", "tfrecords"))
    def test_eval(
        self,
        sota_checkpoints_dir: Path,
        sota_data_dir: Path,
        eval_test_struct: EvalRunParamsStruct,
        dataset_type: str,
        metrics_dump_dir: Path,
        collected_data: List[ResultInfo],
    ):
        if sota_data_dir is None:
            pytest.skip("Path to datasets is not set")
        backend_str = f"TF_{dataset_type}"
        if dataset_type not in eval_test_struct.dataset_types:
            status = f"Skip by: {dataset_type} does not supported for {eval_test_struct.model_name}"
            collected_data.append(
                ResultInfo(
                    model_name=eval_test_struct.model_name,
                    backend=backend_str,
                    status=status,
                )
            )
            pytest.skip(status)

        sample_type = eval_test_struct.sample_type
        metrics_dump_file_path = self.get_metric_file_name(metrics_dump_dir, eval_test_struct.model_name, dataset_type)
        log_dir = metrics_dump_dir / "logs"
        dataset_path = sota_data_dir / dataset_type / eval_test_struct.dataset_name

        weights_param = self.get_weight_params(eval_test_struct, sota_checkpoints_dir)
        cmd = generate_run_examples_command(
            sample_type=sample_type,
            mode="test",
            config=eval_test_struct.config,
            dataset_path=dataset_path,
            dataset_type=dataset_type,
            log_dir=log_dir,
            batch=eval_test_struct.batch,
            metrics_dump_file_path=metrics_dump_file_path,
            **weights_param,
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
                model_name=eval_test_struct.model_name,
                backend=backend_str,
                status=status,
            )
            collected_data.append(result_info)
            pytest.fail(status)

        metric_value = self.read_metric(metrics_dump_file_path)
        fp32_metric = self.get_reference_fp32_metric(metrics_dump_dir, eval_test_struct.reference, dataset_type)

        diff_target = round((metric_value - eval_test_struct.target_tf), 2)
        if fp32_metric:
            diff_fp32 = round((metric_value - fp32_metric), 2)

        threshold_errors = self.threshold_check(
            diff_target=diff_target,
            diff_target_min=eval_test_struct.diff_target_tf_min,
            diff_target_max=eval_test_struct.diff_target_tf_max,
            diff_fp32=diff_fp32,
            diff_fp32_min=eval_test_struct.diff_fp32_min,
            diff_fp32_max=eval_test_struct.diff_fp32_max,
        )

        result_info = ResultInfo(
            model_name=eval_test_struct.model_name,
            backend=backend_str,
            metric_type=eval_test_struct.metric_type,
            measured=metric_value,
            expected=eval_test_struct.target_tf,
            diff_target=diff_target,
            target_fp32=fp32_metric,
            diff_fp32=diff_fp32,
            status=threshold_errors,
        )
        collected_data.append(result_info)
        if threshold_errors is not None:
            pytest.fail(threshold_errors)

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
    @pytest.mark.parametrize("eval_test_struct", EVAL_TEST_STRUCT, ids=idfn)
    def test_openvino_eval(
        self,
        eval_test_struct: EvalRunParamsStruct,
        sota_checkpoints_dir: Path,
        ov_data_dir: Path,
        openvino: bool,
        collected_data: List[ResultInfo],
    ):
        if not openvino:
            pytest.skip()
        if eval_test_struct.skip_ov and (
            eval_test_struct.skip_ov_version is None or eval_test_struct.skip_ov_version == get_openvino_version()
        ):
            status = f"Skip by: {eval_test_struct.skip_ov}"
            collected_data.append(ResultInfo(model_name=eval_test_struct.model_name, backend="OV", status=status))
            pytest.skip(status)

        # WA to avoid OS error
        env = self.get_env()
        env["HDF5_USE_FILE_LOCKING"] = "FALSE"

        sample_type = eval_test_struct.sample_type
        tf_checkpoint = PROJECT_ROOT / "frozen_graph" / f"{eval_test_struct.model_name}.pb"
        ir_model_folder = PROJECT_ROOT / "ir_models" / eval_test_struct.model_name
        ac_config = PROJECT_ROOT / "tests" / "tensorflow" / "data" / "ac_configs" / f"{eval_test_struct.model_name}.yml"
        csv_result = PROJECT_ROOT / f"{eval_test_struct.model_name}.csv"
        csv_result.unlink(missing_ok=True)

        weights_param = self.get_weight_params(eval_test_struct, sota_checkpoints_dir)
        cmd = generate_run_examples_command(
            sample_type=sample_type,
            mode="export",
            config=eval_test_struct.config,
            to_frozen_graph=tf_checkpoint,
            **weights_param,
        )
        runner = Command(cmd, cwd=PROJECT_ROOT, env=env)
        exit_code = runner.run(assert_returncode_zero=False)

        if exit_code != 0:
            status = "ERROR: Failed on export"
            collected_data.append(
                ResultInfo(
                    model_name=eval_test_struct.model_name,
                    backend="OV",
                    status=status,
                )
            )
            pytest.fail(status)
        ov_model = ov.convert_model(tf_checkpoint)
        ir_path = ir_model_folder / f"{eval_test_struct.model_name}.xml"
        ov.serialize(ov_model, ir_path)
        print(ir_path)
        del ov_model

        if eval_test_struct.reference and not ac_config.is_file():
            self.make_ac_config(ac_config, eval_test_struct.reference)

        ac_cmd = (
            f"python -m accuracy_checker.main"
            f" -c {ac_config}"
            f" -s {ov_data_dir}"
            " --progress print"
            f" -d {DATASET_DEFINITIONS_PATH}"
            f" -m {ir_model_folder}"
            f" --csv_result {csv_result}"
        )
        runner = Command(ac_cmd, cwd=PROJECT_ROOT)
        exit_code = runner.run(assert_returncode_zero=False)

        if exit_code != 0:
            status = f"Accuracy checker return code: {exit_code}"
            collected_data.append(
                ResultInfo(
                    model_name=eval_test_struct.model_name,
                    backend="OV",
                    status="ERROR: accuracy_check failed",
                )
            )
            pytest.fail(status)

        metric_value = self.get_metric_from_ac_csv(csv_result)
        fp32_metric = REF_OV_FP32_METRIC.get(eval_test_struct.reference, None)

        diff_target = round((metric_value - eval_test_struct.target_ov), 2)
        diff_fp32 = None
        if fp32_metric:
            diff_fp32 = round((metric_value - fp32_metric), 2)

        threshold_errors = self.threshold_check(
            diff_target=diff_target,
            diff_target_min=eval_test_struct.diff_target_ov_min,
            diff_target_max=eval_test_struct.diff_target_ov_max,
            diff_fp32=diff_fp32,
            diff_fp32_min=eval_test_struct.diff_fp32_min,
            diff_fp32_max=eval_test_struct.diff_fp32_max,
        )
        status = threshold_errors
        if eval_test_struct.xfail_ov is not None and threshold_errors is not None:
            status = f"XFAIL: {eval_test_struct.xfail_ov} {threshold_errors}"

        result_info = ResultInfo(
            model_name=eval_test_struct.model_name,
            backend="OV",
            metric_type=eval_test_struct.metric_type,
            measured=metric_value,
            expected=eval_test_struct.target_ov,
            diff_target=diff_target,
            target_fp32=fp32_metric,
            diff_fp32=diff_fp32,
            status=status,
        )

        collected_data.append(result_info)
        if status is not None:
            if eval_test_struct.xfail_ov is not None:
                pytest.xfail(status)
            else:
                pytest.fail(status)
