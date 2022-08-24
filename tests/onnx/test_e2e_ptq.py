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

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from nncf.common.utils.logger import logger as nncf_logger
from pytest_dependency import depends

TEST_ROOT = Path(__file__).absolute().parents[1]
PROJECT_ROOT = TEST_ROOT.parent.absolute()
BENCHMARKING_DIR = PROJECT_ROOT / 'tests' / "onnx" / "benchmarking"
DATASET_DEFINITIONS_PATH = BENCHMARKING_DIR / "dataset_definitions.yml"

ENV_VARS = os.environ.copy()
if "PYTHONPATH" in ENV_VARS:
    ENV_VARS["PYTHONPATH"] += ":" + str(PROJECT_ROOT)
else:
    ENV_VARS["PYTHONPATH"] = str(PROJECT_ROOT)

MODELS = []
MODELS += [
    ("classification", os.path.splitext(model)[0])
    for model in os.listdir(BENCHMARKING_DIR / "classification" / "onnx_models_configs")
]
MODELS += [
    ("object_detection_segmentation", os.path.splitext(model)[0])
    for model in os.listdir(BENCHMARKING_DIR / "object_detection_segmentation" / "onnx_models_configs")
]

XFAIL_MODELS = {"ssd_mobilenet_v1_12"}

XFAIL_QUANTIZED_MODELS = {
    "shufflenet-9",
    "shufflenet-v2-12",
    "tiny-yolov3-11",
    "yolov3-12",
    "yolov4",
}


def check_xfail(model_name):
    if model_name in XFAIL_MODELS:
        pytest.xfail("ONNXRuntime-OVEP cannot execute the original model")


def check_quantized_xfail(model_name):
    if model_name in XFAIL_QUANTIZED_MODELS:
        pytest.xfail("ONNXRuntime-OVEP cannot execute the quantized model")


def run_command(command):
    with subprocess.Popen(command,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          cwd=PROJECT_ROOT,
                          env=ENV_VARS) as result:
        outs, _ = result.communicate()

        if result.returncode != 0:
            nncf_logger.error(outs.decode("utf-8"))
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
            nncf_logger.info(f"Use anno_dir: {tmp_dir}")
            yield Path(tmp_dir)


@pytest.fixture(scope="module")
def ckpt_dir(request):
    option = request.config.getoption("--ckpt-dir")
    if option is not None:
        yield Path(option)
    else:
        with TemporaryDirectory() as tmp_dir:
            nncf_logger.info(f"Use ckpt_dir: {tmp_dir}")
            yield Path(tmp_dir)


@pytest.fixture(scope="module")
def ptq_size(request):
    return request.config.getoption("--ptq-size")


@pytest.fixture(scope="module")
def eval_size(request):
    option = request.config.getoption("--eval-size")
    if option is None:
        nncf_logger.warning("--eval-size is not provided. Use full dataset for evaluation")
    return option


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
            "-d", str(DATASET_DEFINITIONS_PATH),
            "-m", str(model_dir / task_type / (model_name + ".onnx")),
            "-o", str(ckpt_dir),
            "-s", str(data_dir),
            "-a", str(anno_dir),
            "-ss", str(ptq_size)
        ]

        com_str = ' '.join(com_line)
        nncf_logger.info(f"Run command: {com_str}")
        run_command(com_line)


@pytest.mark.run(order=2)
class TestBenchmark:
    def get_command(
            self, task_type, model_name, model_dir, data_dir, anno_dir, output_dir, eval_size, program, is_quantized):

        program_path = BENCHMARKING_DIR / program

        task_path = BENCHMARKING_DIR / task_type
        config_path = task_path / "onnx_models_configs" / (model_name + ".yml")

        output_dir = output_dir / task_type
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        anno_dir = anno_dir / str(eval_size)
        if not os.path.exists(anno_dir):
            os.makedirs(anno_dir)

        out_file_name = os.path.splitext(program)[0]

        if is_quantized:
            out_file_name += "-quantized.csv"
        else:
            out_file_name += "-original.csv"

        model_file_name = model_name + "-quantized" if is_quantized else model_name
        model_file_name += ".onnx"

        com_line = [
            sys.executable, str(program_path),
            "-c", str(config_path),
            "-d", str(DATASET_DEFINITIONS_PATH),
            "-m", str(model_dir / task_type / model_file_name),
            "-s", str(data_dir),
            "-a", str(anno_dir),
            "--csv_result", str(output_dir / out_file_name)
        ]

        if eval_size is not None:
            com_line += ["-ss", str(eval_size)]

        com_str = ' '.join(com_line)
        nncf_logger.info(f"Run command: {com_str}")
        return com_line

    @pytest.mark.e2e_eval_original_model
    @pytest.mark.parametrize("task_type, model_name", MODELS)
    def test_original_model_accuracy(
            self, task_type, model_name, model_dir, data_dir, anno_dir, output_dir, eval_size):

        check_xfail(model_name)

        command = self.get_command(task_type, model_name, model_dir, data_dir, anno_dir, output_dir, eval_size,
                                   program="accuracy_checker.py", is_quantized=False)
        run_command(command)

    @pytest.mark.e2e_eval_original_model
    @pytest.mark.parametrize("task_type, model_name", MODELS)
    def test_original_model_performance(
            self, task_type, model_name, model_dir, data_dir, anno_dir, output_dir, eval_size):

        check_xfail(model_name)

        command = self.get_command(task_type, model_name, model_dir, data_dir, anno_dir, output_dir, eval_size,
                                   program="performance_checker.py", is_quantized=False)
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

    @pytest.mark.e2e_ptq
    @pytest.mark.dependency()
    @pytest.mark.parametrize("task_type, model_name", MODELS)
    def test_quantized_model_performance(
            self, request, task_type, model_name, ckpt_dir, data_dir, anno_dir, output_dir, eval_size):

        # Run PTQ first
        depends(request, ["TestPTQ::test_ptq_model" + request.node.name.lstrip("test_quantized_model_performance")])
        check_xfail(model_name)
        check_quantized_xfail(model_name)

        model_dir = ckpt_dir
        command = self.get_command(task_type, model_name, model_dir, data_dir, anno_dir, output_dir, eval_size,
                                   program="performance_checker.py", is_quantized=True)
        run_command(command)
