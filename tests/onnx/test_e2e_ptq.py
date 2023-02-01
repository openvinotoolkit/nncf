"""
Copyright (c) 2023 Intel Corporation

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
from typing import List, Dict, Optional

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from yattag import Doc
from yattag import indent

import pytest
from pytest_dependency import depends

from nncf.common.logging.logger import nncf_logger
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
MODELS = [(task, os.path.splitext(model)[0]) for task in TASKS for model in
          os.listdir(BENCHMARKING_DIR / task / "onnx_models_configs")]

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

OV_EP_COL_NAME = "OpenVINOExecutionProvider"
CPU_EP_COL_NAME = "CPUExecutionProvider"
REPORT_NAME = 'report.html'


def check_xfail(model_name):
    if model_name in XFAIL_MODELS:
        pytest.xfail("ONNXRuntime-OVEP cannot execute the reference model")


def check_quantized_xfail(model_name):
    if model_name in XFAIL_QUANTIZED_MODELS:
        pytest.xfail("ONNXRuntime-OVEP cannot execute the quantized model")


def check_skip_model(model_name: str, model_names_to_test: Optional[List[str]]):
    if model_names_to_test is not None and model_name not in model_names_to_test:
        pytest.skip(f'The model {model_name} is skipped, because it was not included in --model-names.')


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
def model_names_to_test(request):
    option = request.config.getoption("--model-names")
    if option is None:
        nncf_logger.info('All models will be tested')
        return option
    option = option.split(' ')
    return option


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


@pytest.fixture(scope="module")
def is_ov_ep(request):
    enable_ov_ep = request.config.getoption("--enable-ov-ep")
    if enable_ov_ep:
        nncf_logger.info("The accuracy validation of the quantized models is enabled for OpenVINOExecutionProvider.")
    else:
        nncf_logger.info("The accuracy validation of the quantized models is disabled for OpenVINOExecutionProvider.")
    return enable_ov_ep


@pytest.fixture(scope="module")
def is_cpu_ep(request):
    disable_cpu_ep = request.config.getoption("--disable-cpu-ep")
    if disable_cpu_ep:
        nncf_logger.info("The accuracy validation of quantized models is disabled for CPUExecutionProvider.")
    else:
        nncf_logger.info("The accuracy validation of quantized models is enabled for CPUExecutionProvider.")
    return not disable_cpu_ep


def _read_accuracy_checker_result(root_dir: Path, key: str) -> pd.DataFrame:
    dfs = []
    for task in TASKS:
        csv_fp = str(root_dir / task / f"accuracy_checker-{key}.csv")
        dfs += [pd.read_csv(csv_fp)]
    df = pd.concat(dfs, axis=0)
    df = df[["model", "metric_value", "metric_name", "tags"]]
    df = df.set_index("model")
    df["model_accuracy"] = df["metric_value"] * 100.0
    df = df[["model_accuracy", "metric_name", "tags"]]
    df = df.pivot_table('model_accuracy', ['model', 'metric_name'], 'tags')
    return df


def _read_reference_json(fpath: Path) -> pd.DataFrame:
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
    df = df[["model", "target_fp32", "target_int8", "metric_type", "diff_target_min", "diff_target_max"]]
    df = df.set_index("model")

    df["model_accuracy"] = round(df["target_fp32"] * 100.0, 3)
    df["metric_name"] = df["metric_type"]
    df = df.drop(columns='metric_type')

    return df


@pytest.fixture
def reference_model_accuracy(scope="module"):
    fpath = ONNX_TEST_ROOT / "data" / "reference_model_accuracy" / "reference.json"

    return _read_reference_json(fpath)


@pytest.fixture
def quantized_model_accuracy(output_dir, scope="function"):
    root_dir = output_dir
    return _read_accuracy_checker_result(root_dir, "quantized")


@pytest.mark.e2e_ptq
@pytest.mark.run(order=1)
class TestPTQ:
    @pytest.mark.dependency()
    @pytest.mark.parametrize("task_type, model_name", MODELS)
    def test_ptq_model(self, task_type, model_name, model_names_to_test, model_dir, data_dir, anno_dir, ckpt_dir,
                       ptq_size):
        check_skip_model(model_name, model_names_to_test)
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
            "-ss", str(ptq_size),
            "--target_tags", CPU_EP_COL_NAME
        ]

        run_command(com_line)


@pytest.mark.run(order=2)
class TestBenchmark:
    @staticmethod
    def get_ac_command(task_type: str, model_name: str, model_dir: Path,
                       data_dir: Path, anno_dir: Path, output_dir: Path,
                       eval_size: int, program: str, is_quantized: bool,
                       is_ov_ep: bool, is_cpu_ep: bool) -> List[str]:

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
        if is_ov_ep and not is_cpu_ep:
            com_line += ["--target_tags", 'OpenVINOExecutionProvider']
        if not is_ov_ep and is_cpu_ep:
            com_line += ["--target_tags", 'CPUExecutionProvider']
        if eval_size is not None:
            com_line += ["-ss", str(eval_size)]

        return com_line

    @pytest.mark.e2e_eval_reference_model
    @pytest.mark.parametrize("task_type, model_name", MODELS)
    def test_reference_model_accuracy(self, task_type, model_name, model_names_to_test, model_dir,
                                      data_dir, anno_dir, output_dir, eval_size):
        check_skip_model(model_name, model_names_to_test)
        check_xfail(model_name)
        # Reference accuracy validation is performed on CPUExecutionProvider
        command = self.get_ac_command(task_type, model_name, model_dir, data_dir, anno_dir, output_dir, eval_size,
                                      program="accuracy_checker.py", is_quantized=False, is_ov_ep=False, is_cpu_ep=True)
        run_command(command)

    @pytest.mark.e2e_ptq
    @pytest.mark.dependency()
    @pytest.mark.parametrize("task_type, model_name", MODELS)
    def test_quantized_model_accuracy(self, request, task_type, model_name, model_names_to_test, ckpt_dir, data_dir,
                                      anno_dir, output_dir, eval_size, is_ov_ep, is_cpu_ep):
        check_skip_model(model_name, model_names_to_test)
        # Run PTQ first
        depends(request, ["TestPTQ::test_ptq_model" + request.node.name.lstrip("test_quantized_model_accuracy")])
        check_xfail(model_name)
        check_quantized_xfail(model_name)

        model_dir = ckpt_dir
        command = self.get_ac_command(task_type, model_name, model_dir, data_dir, anno_dir, output_dir, eval_size,
                                      program="accuracy_checker.py", is_quantized=True, is_ov_ep=is_ov_ep,
                                      is_cpu_ep=is_cpu_ep)
        run_command(command)


@pytest.mark.run(order=3)
class TestBenchmarkResult:
    def join_reference_and_quantized_frames(self, reference_model_accuracy: pd.DataFrame,
                                            quantized_model_accuracy: pd.DataFrame) -> pd.DataFrame:
        df = reference_model_accuracy.join(quantized_model_accuracy)

        df.insert(0, 'Model', '')
        df['Model'] = [df.iloc[i].name[0] for i in range(len(df.index))]
        df = df.reset_index(drop=True)
        df = df.rename({"metric_name": "Metrics type",
                        "model_accuracy": "FP32"}, axis=1)
        if OV_EP_COL_NAME in df.columns:
            df = df.rename({"OpenVINOExecutionProvider": "OV-EP_INT8"}, axis=1)
            df["Diff OV-EP FP32"] = df["OV-EP_INT8"] - df["FP32"]
            df["Diff OV-EP Expected"] = df['target_int8'] * 100 - df["FP32"]
        if CPU_EP_COL_NAME in df.columns:
            df = df.rename({"CPUExecutionProvider": "CPU-EP_INT8"}, axis=1)
            df["Diff CPU-EP FP32"] = df["CPU-EP_INT8"] - df["FP32"]
        df["Expected FP32"] = df["target_fp32"] * 100
        return df

    def get_row_colors(self, df: pd.DataFrame, reference_model_accuracy: pd.DataFrame,
                       int8_col_name: str) -> Dict[int, str]:
        row_colors = {}
        for idx, row in df.iterrows():
            for i, model_name in enumerate(reference_model_accuracy.index):
                if model_name == row['Model']:
                    diff_target_min = reference_model_accuracy.iloc[i]["diff_target_min"]
                    diff_target_max = reference_model_accuracy.iloc[i]["diff_target_max"]
                    target_int8 = reference_model_accuracy.iloc[i]["target_int8"] * 100
                    int8 = row[int8_col_name]
                    fp32 = row["FP32"]
                    if math.isnan(int8):
                        row_colors[idx] = BG_COLOR_RED_HEX
                    elif diff_target_min < int8 - target_int8 < diff_target_max:
                        row_colors[idx] = BG_COLOR_GREEN_HEX
                    elif not diff_target_min < int8 - fp32 < diff_target_max:
                        row_colors[idx] = BG_COLOR_RED_HEX
                    else:
                        row_colors[idx] = BG_COLOR_YELLOW_HEX
                    continue
        return row_colors

    def generate_final_data_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add parentheses, because FP32 metrics were taken from reference.
        df['FP32'] = df['FP32'].astype(str)
        for idx, row in df.iterrows():
            df.at[idx, 'FP32'] = f"({row['FP32']})"
        df = df.fillna("-")
        is_cpu_ep = "CPU-EP_INT8" in df.columns
        is_ov_ep = "OV-EP_INT8" in df.columns

        new_columns_order = ["Model", "Metrics type", "Expected FP32", "FP32"]
        column_names = ["Model", "Metrics type", "Expected FP32", "FP32"]
        if is_cpu_ep:
            new_columns_order.extend(["CPU-EP_INT8", "Diff CPU-EP FP32"])
            column_names.extend(['INT8', 'Diff FP32'])
        if is_ov_ep:
            new_columns_order.extend(["OV-EP_INT8", "Diff OV-EP FP32", "Diff OV-EP Expected"])
            column_names.extend(['INT8', 'Diff FP32', 'Diff Expected'])
        df = df[new_columns_order]
        df.columns = column_names
        if is_cpu_ep and is_ov_ep:
            df.columns = pd.MultiIndex.from_tuples(
                [("", col) for col in df.columns[:4]] + [(CPU_EP_COL_NAME, col) for col in df.columns[4:6]] + [
                    (OV_EP_COL_NAME, col) for col in df.columns[6:]]
            )
            return df
        provider_name = CPU_EP_COL_NAME if is_cpu_ep else OV_EP_COL_NAME
        df.columns = pd.MultiIndex.from_tuples(
            [("", col) for col in df.columns[:4]] + [(provider_name, col) for col in df.columns[4:]]
        )
        return df

    def generate_html(self, df: pd.DataFrame, cpu_ep_row_colors: Dict[int, str], ov_ep_row_colors: Dict[int, str],
                      output_fp: str) -> None:
        doc, tag, text = Doc().tagtext()
        doc.asis('<!DOCTYPE html>')
        with tag('head'):
            with tag('style'):
                doc.asis("green_text" + "{Background-color: " + f"#{BG_COLOR_GREEN_HEX}" + "}")
                doc.asis("yellow_text" + "{Background-color: " + f"#{BG_COLOR_YELLOW_HEX}" + "}")
                doc.asis("red_text" + "{Background-color: " + f"#{BG_COLOR_RED_HEX}" + "}")
                doc.asis("report_table" + " border-collapse: collapse; border: 1px solid;")
        with tag('p'):
            text('legend: ')
        with tag('p'):
            with tag('green_text'):
                text('Thresholds for FP32 and Expected are passed')
        with tag('p'):
            with tag('yellow_text'):
                text('Thresholds for Expected is failed, but for FP32 passed')
        with tag('p'):
            with tag('red_text'):
                text('Thresholds for FP32 and Expected are failed')
        with tag('p'):
            text('If Reference FP32 value in parentheses, it takes from "target" field of .json file')
        with tag('report_table'):
            with tag('table', border="1", cellpadding="5"):
                # First row with merging cells with the same name
                with tag('tr'):
                    prev_el, _ = df.columns[0]
                    cnt = 1
                    for up_col, _ in df.columns[1:]:
                        if up_col != prev_el:
                            with tag('td', colspan=cnt):
                                text(prev_el)
                            cnt = 0
                        prev_el = up_col
                        cnt += 1
                    with tag('td', colspan=cnt):
                        text(prev_el)
                # Second row
                with tag('tr'):
                    for _, bot_col in df.columns:
                        with tag('td'):
                            text(bot_col)
                # Data cells
                with tag('tr'):
                    for idx, row in df.iterrows():
                        with tag('tr'):
                            for i, elem in enumerate(row):
                                additional_attrs = {}
                                up_col, bot_col = df.columns[i]
                                if up_col == OV_EP_COL_NAME:
                                    additional_attrs = {'bgcolor': f'{ov_ep_row_colors[idx]}'}
                                elif up_col == CPU_EP_COL_NAME:
                                    additional_attrs = {'bgcolor': f'{cpu_ep_row_colors[idx]}'}
                                with tag('td', **additional_attrs):
                                    if isinstance(elem, float):
                                        elem = round(elem, 2)
                                    text(elem)

        with open(output_fp, 'w', encoding='utf8') as f:
            f.write(indent(doc.getvalue(), indent_text=True))

    @pytest.mark.e2e_ptq
    @pytest.mark.dependency()
    @pytest.mark.parametrize("task_type, model_name", MODELS)
    def test_model_accuracy(self, request, task_type, model_name, model_names_to_test, reference_model_accuracy,
                            quantized_model_accuracy):
        check_skip_model(model_name, model_names_to_test)
        # Run PTQ first
        depends(request, ["TestPTQ::test_ptq_model" + request.node.name.lstrip("test_quantized_model_performance")])
        check_xfail(model_name)
        check_quantized_xfail(model_name)

        df = self.join_reference_and_quantized_frames(reference_model_accuracy, quantized_model_accuracy)
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
        output_fp = str(output_dir / REPORT_NAME)
        cpu_ep_row_colors, ov_ep_row_colors = {}, {}
        is_ov_ep = OV_EP_COL_NAME in quantized_model_accuracy.columns
        is_cpu_ep = CPU_EP_COL_NAME in quantized_model_accuracy.columns
        df = self.join_reference_and_quantized_frames(reference_model_accuracy, quantized_model_accuracy)
        if is_cpu_ep:
            cpu_ep_row_colors = self.get_row_colors(df, reference_model_accuracy, "CPU-EP_INT8")
        if is_ov_ep:
            ov_ep_row_colors = self.get_row_colors(df, reference_model_accuracy, "OV-EP_INT8")
        df = self.generate_final_data_frame(df)
        self.generate_html(df, cpu_ep_row_colors, ov_ep_row_colors, output_fp)
