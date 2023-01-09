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
import yaml

import pytest
from pytest_dependency import depends

from nncf.common.logging import nncf_logger
from tests.shared.paths import PROJECT_ROOT
from tests.shared.command import Command
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

POT_XFAIL_QUANTIZED_MODELS = {
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
POT_EP_COL_NAME = "POT"
REPORT_NAME = 'report.html'


def check_xfail(model_name):
    if model_name in XFAIL_MODELS:
        pytest.xfail("ONNXRuntime-OVEP cannot execute the reference model")


def check_quantized_xfail(model_name):
    if model_name in XFAIL_QUANTIZED_MODELS:
        pytest.xfail("ONNXRuntime-OVEP cannot execute the quantized model")


def check_pot_quantized_xfail(model_name):
    if model_name in POT_XFAIL_QUANTIZED_MODELS:
        pytest.xfail("POT can not quantize the model")


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
    disable_ov_ep = request.config.getoption("--disable-ov-ep")
    if disable_ov_ep:
        nncf_logger.info("The accuracy validation of the quantized models is disabled for OpenVINOExecutionProvider.")
    else:
        nncf_logger.info("The accuracy validation of the quantized models is enabled for OpenVINOExecutionProvider.")
    return not disable_ov_ep


@pytest.fixture(scope="module")
def is_cpu_ep(request):
    enable_cpu_ep = request.config.getoption("--enable-cpu-ep")
    if enable_cpu_ep:
        nncf_logger.info("The accuracy validation of quantized models is enabled for CPUExecutionProvider.")
    else:
        nncf_logger.info("The accuracy validation of quantized models is disabled for CPUExecutionProvider.")
    return enable_cpu_ep


@pytest.fixture(scope="module")
def is_pot(request):
    enable_pot = request.config.getoption("--enable-pot")
    if enable_pot:
        nncf_logger.info("The quantization by POT is enabled.")
    return enable_pot


def _read_accuracy_checker_result(root_dir: Path, key: str, is_pot: bool) -> pd.DataFrame:
    dfs = []
    for task in TASKS:
        csv_fp = str(root_dir / task / f"accuracy_checker-{key}.csv")
        dfs += [pd.read_csv(csv_fp)]

    df = pd.concat(dfs, axis=0)
    additional_column = 'tags' if not is_pot else 'device'
    df = df[["model", "metric_value", "metric_name", additional_column]]
    df = df.set_index("model")
    df["model_accuracy"] = df["metric_value"] * 100.0
    df = df[["model_accuracy", "metric_name", additional_column]]
    df = df.pivot_table('model_accuracy', ['model', 'metric_name'], additional_column)
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


def modify_ac_config(config_path, data_dir, anno_dir):
    data = None
    with open(config_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.loader.SafeLoader)
        data['models'][0]['datasets'][0]['data_source'] = str(
            data_dir / Path(data['models'][0]['datasets'][0]['data_source']))
        data['models'][0]['datasets'][0]['annotation_conversion']['annotation_file'] = str(
            data_dir / Path(data['models'][0]['datasets'][0]['annotation_conversion']['annotation_file']))
        data['models'][0]['datasets'][0]['annotation'] = str(
            anno_dir / Path(data['models'][0]['datasets'][0]['annotation']))
    with open(config_path, 'w') as f:
        f.write(yaml.dump(data, default_flow_style=False))


@pytest.fixture
def reference_model_accuracy(scope="module"):
    fpath = ONNX_TEST_ROOT / "data" / "reference_model_accuracy" / "reference.json"

    return _read_reference_json(fpath)


@pytest.fixture
def quantized_model_accuracy(output_dir, scope="function"):
    root_dir = output_dir
    return _read_accuracy_checker_result(root_dir, "quantized", is_pot=False)


@pytest.fixture
def quantized_pot_model_accuracy(output_dir, is_pot, scope="function"):
    if not is_pot:
        return None
    root_dir = output_dir
    return _read_accuracy_checker_result(root_dir, "pot-quantized", is_pot)


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
            "-ss", str(ptq_size)
        ]

        com_str = ' '.join(com_line)
        run_command(com_line)

    def get_ir_model(self, model_path, model_name, output_dir):
        runner = Command(f"mo -m {model_path} -o {output_dir} -n {model_name}")
        runner.run()

    def get_quantized_pot_model(self, model_dir, model_name, config_path):
        model_topology = str(model_dir / model_name) + '.xml'
        model_weights = str(model_dir / model_name) + '.bin'
        output_pot_model_dir = str(model_dir / model_name)
        runner = Command(f'pot -q default -m {model_topology} -w {model_weights} \
         --ac-config {config_path} --engine accuracy_checker --output-dir {output_pot_model_dir} --name {model_name} --direct-dump')
        runner.run()

    @pytest.mark.dependency()
    @pytest.mark.parametrize("task_type, model_name", MODELS)
    def test_pot_model(self, task_type, model_name, model_dir, model_names_to_test, data_dir, anno_dir, ckpt_dir,
                       ptq_size):
        check_skip_model(model_name, model_names_to_test)
        check_pot_quantized_xfail(model_name)

        task_path = BENCHMARKING_DIR / task_type
        config_path = task_path / "openvino_models_configs" / (model_name + ".yml")

        ckpt_dir = ckpt_dir / task_type
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        anno_dir = anno_dir / str(ptq_size)
        if not os.path.exists(anno_dir):
            os.makedirs(anno_dir)

        model_path = model_dir / task_type / (model_name + ".onnx")
        ir_model_dir = ckpt_dir / 'openvino'
        self.get_ir_model(model_path, model_name, ir_model_dir)
        modify_ac_config(config_path, data_dir, anno_dir)
        self.get_quantized_pot_model(ir_model_dir, model_name, config_path)


@pytest.mark.run(order=2)
class TestBenchmark:
    @staticmethod
    def get_ac_command(task_type: str, model_name: str, model_dir: Path,
                       data_dir: Path, anno_dir: Path, output_dir: Path,
                       eval_size: int, program: str, is_quantized: bool,
                       is_ov_ep: bool, is_cpu_ep: bool, is_pot: bool) -> List[str]:

        program_path = BENCHMARKING_DIR / program
        task_path = BENCHMARKING_DIR / task_type
        anno_dir = anno_dir / str(eval_size)
        if not anno_dir.exists():
            anno_dir.mkdir(parents=True)

        config_path = task_path / "onnx_models_configs" / (model_name + ".yml")
        if is_pot:
            config_path = task_path / "openvino_models_configs" / (model_name + ".yml")
            modify_ac_config(config_path, data_dir, anno_dir)

        output_dir = output_dir / task_type
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        out_file_name = os.path.splitext(program)[0]

        if is_quantized:
            if is_pot:
                out_file_name += '-pot-quantized.csv'
            else:
                out_file_name += "-quantized.csv"
        else:
            out_file_name += "-reference.csv"

        model_file_name = model_name + "-quantized" if is_quantized else model_name
        model_file_name += ".onnx"
        if is_pot:
            model_file_name = model_dir / task_type / Path('openvino') / Path(model_name) / Path('optimized')

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
                                      anno_dir, output_dir,
                                      eval_size, is_ov_ep, is_cpu_ep):
        check_skip_model(model_name, model_names_to_test)
        # Run PTQ first
        depends(request, ["TestPTQ::test_ptq_model" + request.node.name.lstrip("test_quantized_model_accuracy")])
        check_xfail(model_name)
        check_quantized_xfail(model_name)

        model_dir = ckpt_dir
        command = self.get_ac_command(task_type, model_name, model_dir, data_dir, anno_dir, output_dir, eval_size,
                                      program="accuracy_checker.py", is_quantized=True, is_ov_ep=is_ov_ep,
                                      is_cpu_ep=is_cpu_ep, is_pot=False)
        run_command(command)

    @pytest.mark.e2e_ptq
    @pytest.mark.dependency()
    @pytest.mark.parametrize("task_type, model_name", MODELS)
    def test_quantized_pot_model_accuracy(self, request, task_type, model_name, model_names_to_test, ckpt_dir, data_dir,
                                          anno_dir, output_dir,
                                          eval_size):
        check_skip_model(model_name, model_names_to_test)
        # Run POT first
        depends(request, ["TestPTQ::test_pot_model" + request.node.name.lstrip("test_quantized_pot_model_accuracy")])
        check_xfail(model_name)
        check_quantized_xfail(model_name)

        model_dir = ckpt_dir
        command = self.get_ac_command(task_type, model_name, model_dir, data_dir, anno_dir, output_dir, eval_size,
                                      program="accuracy_checker.py", is_quantized=True, is_ov_ep=False,
                                      is_cpu_ep=False, is_pot=True)
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
        if 'CPU' in df.columns:
            # POT
            df = df.rename({"CPU": "POT_INT8"}, axis=1)
            df["Diff POT FP32"] = df["POT_INT8"] - df["FP32"]
        if OV_EP_COL_NAME in df.columns:
            df = df.rename({"OpenVINOExecutionProvider": "OV-EP_INT8"}, axis=1)
            df["Diff OV-EP FP32"] = df["OV-EP_INT8"] - df["FP32"]
            df["Expected FP32"] = df["target_fp32"] * 100
            df["Diff OV-EP Expected"] = df['target_int8'] * 100 - df["FP32"]
        if CPU_EP_COL_NAME in df.columns:
            df = df.rename({"CPUExecutionProvider": "CPU-EP_INT8"}, axis=1)
            df["Diff CPU-EP FP32"] = df["CPU-EP_INT8"] - df["FP32"]
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
                    elif int8 - fp32 > -1.0:
                        row_colors[idx] = BG_COLOR_GREEN_HEX
                    else:
                        row_colors[idx] = BG_COLOR_RED_HEX
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
        is_pot = 'POT_INT8' in df.columns
        new_columns_order = ["Model", "Metrics type", "Expected FP32", "FP32"]
        column_names = ["Model", "Metrics type", "Expected FP32", "FP32"]
        if is_cpu_ep:
            new_columns_order.extend(["CPU-EP_INT8", "Diff CPU-EP FP32"])
            column_names.extend(['INT8', 'Diff FP32'])
        if is_ov_ep:
            new_columns_order.extend(["OV-EP_INT8", "Diff OV-EP FP32", "Diff OV-EP Expected"])
            column_names.extend(['INT8', 'Diff FP32', 'Diff Expected'])
        if is_pot:
            new_columns_order.extend(["POT_INT8", "Diff POT FP32"])
            column_names.extend(['INT8', 'Diff FP32'])
        df = df[new_columns_order]
        df.columns = column_names
        columns = [("", col) for col in df.columns[:4]]
        if is_cpu_ep:
            columns += [(CPU_EP_COL_NAME, col) for col in df.columns[4:6]]
        if is_ov_ep and is_cpu_ep:
            columns += [(OV_EP_COL_NAME, col) for col in df.columns[6:9]]
        elif is_ov_ep:
            columns += [(OV_EP_COL_NAME, col) for col in df.columns[4:7]]
        if is_pot:
            columns += [(POT_EP_COL_NAME, col) for col in df.columns[-2:]]
        df.columns = pd.MultiIndex.from_tuples(columns)
        return df

    def generate_html(self, df: pd.DataFrame, cpu_ep_row_colors: Dict[int, str], ov_ep_row_colors: Dict[int, str],
                      pot_row_colors: Dict[int, str],
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
                                elif up_col == POT_EP_COL_NAME:
                                    additional_attrs = {'bgcolor': f'{pot_row_colors[idx]}'}
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
    def test_generate_report(self, reference_model_accuracy, quantized_model_accuracy, quantized_pot_model_accuracy,
                             output_dir):
        output_fp = str(output_dir / REPORT_NAME)
        cpu_ep_row_colors, ov_ep_row_colors, pot_row_colors = {}, {}, {}
        is_ov_ep = OV_EP_COL_NAME in quantized_model_accuracy.columns
        is_cpu_ep = CPU_EP_COL_NAME in quantized_model_accuracy.columns
        is_pot = False
        if quantized_pot_model_accuracy is not None:
            is_pot = True
            quantized_model_accuracy = quantized_model_accuracy.join(quantized_pot_model_accuracy)
        df = self.join_reference_and_quantized_frames(reference_model_accuracy, quantized_model_accuracy)
        if is_cpu_ep:
            cpu_ep_row_colors = self.get_row_colors(df, reference_model_accuracy, "CPU-EP_INT8")
        if is_ov_ep:
            ov_ep_row_colors = self.get_row_colors(df, reference_model_accuracy, "OV-EP_INT8")
        if is_pot:
            pot_row_colors = self.get_row_colors(df, reference_model_accuracy, "POT_INT8")
        df = self.generate_final_data_frame(df)
        self.generate_html(df, cpu_ep_row_colors, ov_ep_row_colors, pot_row_colors, output_fp)
