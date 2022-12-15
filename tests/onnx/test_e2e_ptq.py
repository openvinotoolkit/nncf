"""
Copyright (c) 2022 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=redefined-outer-name

import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import pandas as pd
import pytest
from pytest_dependency import depends

from tests.shared.paths import PROJECT_ROOT
from tests.onnx.conftest import ONNX_TEST_ROOT

BG_COLOR_GREEN_HEX = 'ccffcc'
BG_COLOR_YELLOW_HEX = 'ffffcc'
BG_COLOR_RED_HEX = 'ffcccc'

BENCHMARKING_DIR = ONNX_TEST_ROOT / "benchmarking"

ENV_VARS = os.environ.copy()
if "PYTHONPATH" in ENV_VARS:
    ENV_VARS["PYTHONPATH"] += ":" + str(PROJECT_ROOT)
else:
    ENV_VARS["PYTHONPATH"] = str(PROJECT_ROOT)

TASKS = ["classification", "object_detection_segmentation"]
MODELS = list(itertools.chain(*[
    [(task, os.path.splitext(model)[0])
     for model in os.listdir(BENCHMARKING_DIR / task / "onnx_models_configs")]
    for task in TASKS]))

XFAIL_MODELS = {}

XFAIL_QUANTIZED_MODELS = {
}

# TODO(vshampor): Somehow installing onnxruntime-openvino does not install the OV package in the way
#  that we are used to, and accuracy checker invocations cannot find the dataset_definitions.yml
#  file using the regular path pointing to site-packages, hence the need for using a copy of
#  a hopefully relevant dataset_definitions.yml taken from the tests dir. Should investigate
#  if onnxruntime-openvino actually has a relevant dataset_definitions.yml file somewhere within its own
#  site-packages directory.
DATASET_DEFINITIONS_PATH_ONNX = BENCHMARKING_DIR / 'dataset_definitions.yml'

def check_xfail(model_name):
    if model_name in XFAIL_MODELS:
        pytest.xfail("ONNXRuntime-OVEP cannot execute the reference model")


def check_quantized_xfail(model_name):
    if model_name in XFAIL_QUANTIZED_MODELS:
        pytest.xfail("ONNXRuntime-OVEP cannot execute the quantized model")


def run_command(command: List[str]):
    com_str = ' '.join(command)
    print(f"Run command: {com_str}")
    with subprocess.Popen(command,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          cwd=PROJECT_ROOT,
                          env=ENV_VARS) as result:
        outs, _ = result.communicate()

        if result.returncode != 0:
            print(outs.decode("utf-8"))
            pytest.fail()


@pytest.fixture(scope="module")
def model_dir(request):
    option = request.config.getoption("--model-dir")
    if option is None:
        pytest.skip(f"--model-dir option is required to run {request.node.name}")
    return Path(option)


@pytest.fixture(scope="module")
def data_dir(request):
    option = request.config.getoption("--data-dir")
    if option is None:
        pytest.skip(f"--data-dir option is required to run {request.node.name}")
    return Path(option)


@pytest.fixture(scope="module")
def output_dir(request):
    option = request.config.getoption("--output-dir")
    if option is None:
        pytest.skip(f"--output-dir option is required to run {request.node.name}")
    return Path(option)


@pytest.fixture(scope="module")
def anno_dir(request):
    option = request.config.getoption("--anno-dir")
    if option is not None:
        yield Path(option)
    else:
        with TemporaryDirectory() as tmp_dir:
            print(f"Use anno_dir: {tmp_dir}")
            yield Path(tmp_dir)


@pytest.fixture(scope="module")
def ckpt_dir(request):
    option = request.config.getoption("--ckpt-dir")
    if option is not None:
        yield Path(option)
    else:
        with TemporaryDirectory() as tmp_dir:
            print(f"Use ckpt_dir: {tmp_dir}")
            yield Path(tmp_dir)


@pytest.fixture(scope="module")
def ptq_size(request):
    return request.config.getoption("--ptq-size")


@pytest.fixture(scope="module")
def eval_size(request):
    option = request.config.getoption("--eval-size")
    if option is None:
        print("--eval-size is not provided. Use full dataset for evaluation")
    return option


def _read_csv(root_dir: Path, key: str) -> pd.DataFrame:
    dfs = []
    for task in TASKS:
        csv_fp = str(root_dir / task / f"accuracy_checker-{key}.csv")
        dfs += [pd.read_csv(csv_fp)]
    df = pd.concat(dfs, axis=0)
    df = df[["model", "metric_value", "metric_name"]]
    df = df.set_index("model")
    df["model_accuracy"] = df["metric_value"] * 100.0
    df = df[["model_accuracy", "metric_name"]]
    return df


def _read_json(fpath: Path) -> pd.DataFrame:
    fpath = str(fpath)
    with open(fpath, "r", encoding="utf-8") as fp:
        d0 = json.load(fp)

    rows = []

    for task, d1 in d0.items():
        for dataset, d2 in d1.items():
            for model, d3 in d2.items():
                d3["task"] = task
                d3["dataset"] = dataset
                d3["model"] = model
                row = pd.Series(d3)
                rows += [row]

    df = pd.DataFrame(rows)
    df = df[["model", "target", "metric_type", "diff_target_min", "diff_target_max"]]
    df = df.set_index("model")

    df["model_accuracy"] = df["target"] * 100.0
    df["metric_name"] = df["metric_type"]

    return df


@pytest.fixture
def reference_model_accuracy(scope="module"):
    fpath = ONNX_TEST_ROOT / "data" / "reference_model_accuracy" / "reference.json"

    return _read_json(fpath)


@pytest.fixture
def quantized_model_accuracy(output_dir, scope="function"):
    root_dir = output_dir
    return _read_csv(root_dir, "quantized")


@pytest.mark.e2e_ptq
@pytest.mark.run(order=1)
class TestPTQ:
    @pytest.mark.dependency()
    @pytest.mark.parametrize("task_type, model_name", MODELS)
    def test_ptq_model(self, task_type, model_name, model_dir, data_dir, anno_dir, ckpt_dir, ptq_size):
        check_xfail(model_name)

        program_path = BENCHMARKING_DIR / "run_ptq.py"

        task_path = BENCHMARKING_DIR / task_type
        config_path = task_path / "onnx_models_configs" / (model_name + ".yml")

        ckpt_dir = ckpt_dir / task_type
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        anno_dir = anno_dir / str(ptq_size)
        if not os.path.exists(anno_dir):
            os.makedirs(anno_dir)

        com_line = [
            sys.executable, str(program_path),
            "-c", str(config_path),
            "-d", str(DATASET_DEFINITIONS_PATH_ONNX),
            "-m", str(model_dir / task_type / (model_name + ".onnx")),
            "-o", str(ckpt_dir),
            "-s", str(data_dir),
            "-a", str(anno_dir),
            "-ss", str(ptq_size)
        ]

        com_str = ' '.join(com_line)
        print(f"Run command: {com_str}")
        run_command(com_line)


@pytest.mark.run(order=2)
class TestBenchmark:
    @staticmethod
    def get_command(
            task_type: str,
            model_name: str,
            model_dir: Path,
            data_dir: Path,
            anno_dir: Path,
            output_dir: Path,
            eval_size: int, program: str, is_quantized: bool) -> List[str]:

        program_path = BENCHMARKING_DIR / program

        task_path = BENCHMARKING_DIR / task_type
        config_path = task_path / "onnx_models_configs" / (model_name + ".yml")

        output_dir = output_dir / task_type
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        anno_dir = anno_dir / str(eval_size)
        if not anno_dir.exists():
            anno_dir.mkdir(parents=True)

        out_file_name = os.path.splitext(program)[0]

        if is_quantized:
            out_file_name += "-quantized.csv"
        else:
            out_file_name += "-reference.csv"

        model_file_name = model_name + "-quantized" if is_quantized else model_name
        model_file_name += ".onnx"

        com_line = [
            sys.executable, str(program_path),
            "-c", str(config_path),
            "-d", str(DATASET_DEFINITIONS_PATH_ONNX),
            "-m", str(model_dir / task_type / model_file_name),
            "-s", str(data_dir),
            "-a", str(anno_dir),
            "--csv_result", str(output_dir / out_file_name)
        ]

        if eval_size is not None:
            com_line += ["-ss", str(eval_size)]

        return com_line

    @pytest.mark.e2e_eval_reference_model
    @pytest.mark.parametrize("task_type, model_name", MODELS)
    def test_reference_model_accuracy(
            self, task_type, model_name, model_dir, data_dir, anno_dir, output_dir, eval_size):

        check_xfail(model_name)

        command = self.get_command(task_type, model_name, model_dir, data_dir, anno_dir, output_dir, eval_size,
                                   program="accuracy_checker.py", is_quantized=False)
        run_command(command)

    @pytest.mark.e2e_ptq
    @pytest.mark.dependency()
    @pytest.mark.parametrize("task_type, model_name", MODELS)
    def test_quantized_model_accuracy(
            self, request, task_type, model_name, ckpt_dir, data_dir, anno_dir, output_dir, eval_size):

        # Run PTQ first
        depends(request, ["TestPTQ::test_ptq_model" + request.node.name.lstrip("test_quantized_model_accuracy")])
        check_xfail(model_name)
        check_quantized_xfail(model_name)

        model_dir = ckpt_dir
        command = self.get_command(task_type, model_name, model_dir, data_dir, anno_dir, output_dir, eval_size,
                                   program="accuracy_checker.py", is_quantized=True)
        run_command(command)


@pytest.mark.run(order=3)
class TestBenchmarkResult:
    def parse_df(self, reference_model_accuracy, quantized_model_accuracy):
        df = reference_model_accuracy.join(quantized_model_accuracy, lsuffix="_FP32", rsuffix="_INT8")

        df = df.reset_index()
        df = df.rename({"model": "Model", "metric_name_FP32": "Metrics type",
                        "model_accuracy_FP32": "FP32", "model_accuracy_INT8": "INT8",
                        "diff_target_min_FP32": "diff_target_min",
                        "diff_target_max_FP32": "diff_target_max"}, axis=1)

        df["Diff FP32"] = df["INT8"] - df["FP32"]
        # TODO: Change E2E test to make "Expected FP32" column effective.
        df["Expected FP32"] = df["FP32"]
        df["Diff Expected"] = df["INT8"] - df["Expected FP32"]

        return df

    @pytest.mark.e2e_ptq
    @pytest.mark.dependency()
    @pytest.mark.parametrize("task_type, model_name", MODELS)
    def test_model_accuracy(self, request, task_type, model_name, reference_model_accuracy, quantized_model_accuracy):
        # Run PTQ first
        depends(request, ["TestPTQ::test_ptq_model" + request.node.name.lstrip("test_quantized_model_performance")])
        check_xfail(model_name)
        check_quantized_xfail(model_name)

        df = self.parse_df(reference_model_accuracy, quantized_model_accuracy)
        df = df.set_index("Model")
        this_model_accuracy = df[df.index.str.contains(model_name)]

        assert len(this_model_accuracy) > 0, f"{model_name} has no result from the table."

        for index, cols in this_model_accuracy.iterrows():
            min_threshold = cols["diff_target_min"]
            max_threshold = cols["diff_target_max"]
            assert min_threshold < cols["Diff FP32"] < max_threshold, \
                f"Diff Expected of {index} should be in ({min_threshold:.1f}%, {max_threshold:.1f}%)."

    @pytest.mark.e2e_ptq
    @pytest.mark.run(order=4)
    def test_generate_report(self, reference_model_accuracy, quantized_model_accuracy, output_dir):
        output_fp = str(output_dir / "report.html")

        df = self.parse_df(reference_model_accuracy, quantized_model_accuracy)

        yellow_rows = []
        red_rows = []
        green_rows = []

        for idx, row in df.iterrows():
            if math.isnan(row["INT8"]):
                red_rows += [idx]
            elif row["diff_target_min"] < row["Diff FP32"] < row["diff_target_max"]:
                green_rows += [idx]
            else:
                yellow_rows += [idx]

        df = df[["Model", "Metrics type", "Expected FP32", "FP32", "INT8", "Diff FP32", "Diff Expected"]]
        # Add ONNXRuntime-OpenVINOExecutionProvider multi-column on the top of 3 ~ 6 columns
        hier_col_name = "ONNXRuntime-OpenVINOExecutionProvider"
        df.columns = pd.MultiIndex.from_tuples(
            [("", col) for col in df.columns[:3]] + [(hier_col_name, col) for col in df.columns[3:]]
        )

        def _style_rows():
            styles = []
            # 3 ~ 6 columns are allowed to be colored.

            for col in range(3, 7):
                for idx in yellow_rows:
                    styles.append(f"""
                    .row{idx}.col{col} {{background-color: #{BG_COLOR_YELLOW_HEX};}}
                    """)
                for idx in red_rows:
                    styles.append(f"""
                    .row{idx}.col{col} {{background-color: #{BG_COLOR_RED_HEX};}}
                    """)
                for idx in green_rows:
                    styles.append(f"""
                    .row{idx}.col{col} {{background-color: #{BG_COLOR_GREEN_HEX};}}
                    """)

            return "\n".join(styles)

        legend_info = f"""
        <p>legend:</p>
        <p>
            <span style='Background-color: #{BG_COLOR_GREEN_HEX}'>
                Thresholds for FP32 and Expected are passed
            </span>
        </p>
        <p>
            <span style='Background-color: #{BG_COLOR_YELLOW_HEX}'>
                Thresholds for Expected is failed, but for FP32 passed
            </span>
        </p>
        <p>
            <span style='Background-color: #{BG_COLOR_RED_HEX}'>
                Thresholds for FP32 and Expected are failed
            </span>
        </p>
        <p>
            If Reference FP32 value in parentheses, it takes from "target" field of .json file
        </p>
        """
        # Replace NaN values with "-"
        df = df.fillna("-")

        with open(output_fp, "w", encoding="utf-8") as fp:
            fp.write(
f"""
<html>
<head>
<style>
table, th, td {{font-size:10pt; border:1px solid black; border-collapse:collapse; text-align:center;}}
th, td {{padding: 5px; }}
{_style_rows()}
</style>
</head>
<body>
{legend_info}
{df.style.format({(hier_col_name, "FP32"): "({:.2f})"}).set_precision(2).render()}
</body>
</html>
""")
