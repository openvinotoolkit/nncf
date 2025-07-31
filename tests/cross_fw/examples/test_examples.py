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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from tests.cross_fw.shared.case_collection import skip_if_backend_not_selected
from tests.cross_fw.shared.command import Command
from tests.cross_fw.shared.helpers import create_venv_with_nncf
from tests.cross_fw.shared.helpers import get_pip_executable_with_venv
from tests.cross_fw.shared.helpers import get_python_executable_with_venv
from tests.cross_fw.shared.json import load_json
from tests.cross_fw.shared.paths import PROJECT_ROOT
from tests.cross_fw.shared.paths import TEST_ROOT

EXAMPLE_TEST_ROOT = TEST_ROOT / "cross_fw" / "examples"
EXAMPLE_SCOPE_PATH = EXAMPLE_TEST_ROOT / "example_scope.json"

ACCURACY_TOLERANCE = 0.002
PERFORMANCE_RELATIVE_TOLERANCE = 0.05
MODEL_SIZE_RELATIVE_TOLERANCE = 0.05

ACCURACY_METRICS = "accuracy_metrics"
ACCURACY_METRICS_AFTER_TRAINING = "accuracy_metrics_after_training"
MODEL_SIZE_METRICS = "model_size_metrics"
PERFORMANCE_METRICS = "performance_metrics"

NUM_RETRY_ON_CONNECTION_ERROR = 2
RETRY_TIMEOUT = 60

def example_test_cases():
    example_scope = load_json(EXAMPLE_SCOPE_PATH)
    for example_name, example_params in example_scope.items():
        marks = pytest.mark.cuda if example_params.get("device") == "cuda" else ()
        yield pytest.param(example_name, example_params, id=example_name, marks=marks)


def _is_connection_error(txt: str) -> bool:
    error_list = [
        "ReadTimeoutError",
        "HTTPError",
        "URL fetch failure",
    ]
    for line in txt.split()[::-1]:
        if any(e in line for e in error_list):
            print("-------------------------------")
            print(f"Detect connection error: {line}")
            return True
    return False


@pytest.mark.parametrize("example_name, example_params", example_test_cases())
def test_examples(
    tmp_path: Path,
    example_name: str,
    example_params: dict[str, Any],
    backends_list: list[str],
    is_check_performance: bool,
    ov_version_override: str,
    data: str,
    reuse_venv: bool,
):
    print("\n" + "-" * 64)
    print(f"Example name: {example_name}")
    python_version = sys.version_info
    example_python_version = tuple(example_params.get("python_version", python_version))
    if python_version < example_python_version:
        pytest.skip(f"The test is skipped because python >= {example_python_version} is required.")
    backend = example_params["backend"]
    device = example_params.get("device")
    skip_if_backend_not_selected(backend, backends_list)
    if reuse_venv:
        # Use example directory as tmp_path
        tmp_path = (PROJECT_ROOT / example_params["requirements"]).parent
    venv_path = create_venv_with_nncf(tmp_path, "pip_e_local", "venv", {})
    pip_with_venv = get_pip_executable_with_venv(venv_path)
    install_wwb = False
    if "requirements" in example_params:
        requirements = PROJECT_ROOT / example_params["requirements"]

        # Install whowhatbench later only if
        # explicitly required by the example
        with open(requirements, 'r', encoding='utf-8') as f:
            requirements_content = f.read()
            if 'whowhatbench' in requirements_content:
                install_wwb = True

        run_cmd_line = f"{pip_with_venv} install -r {requirements}"
        print(f"Installing requirements: {run_cmd_line}")
        subprocess.run(run_cmd_line, check=True, shell=True)

    if ov_version_override is not None:
        ov_version_cmd_line = f"{pip_with_venv} install {ov_version_override}"
        uninstall_cmd_line = f"{pip_with_venv} uninstall --yes openvino-genai openvino_tokenizers"
        extra_index_url = "https://storage.openvinotoolkit.org/simple/wheels/nightly"
        print(f"Installing OpenVINO version override: {ov_version_cmd_line}")
        subprocess.run(ov_version_cmd_line, check=True, shell=True)

        if install_wwb:
            wwb_module_string = "whowhatbench@git+https://github.com/openvinotoolkit/openvino.genai.git#subdirectory=tools/who_what_benchmark"
            wwb_override_cmd_line = f"{pip_with_venv} install --pre --extra-index-url {extra_index_url} {wwb_module_string}"
            print(f"Uninstalling OpenVINO packages: {uninstall_cmd_line}")
            subprocess.run(uninstall_cmd_line, check=True, shell=True)
            print(f"Installing WWB module: {wwb_override_cmd_line}")
            subprocess.run(wwb_override_cmd_line, check=True, shell=True)

    cmd_list_packages = f"{pip_with_venv} list"
    print(f"Listing installed packages: {cmd_list_packages}")
    subprocess.run(cmd_list_packages, check=True, shell=True)

    env = os.environ.copy()
    example_dir = Path(example_params["requirements"]).parent
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{example_dir}"  # need this to be able to import from tests.* in run_example.py
    )
    env["ONEDNN_MAX_CPU_ISA"] = "AVX2"  # Set ISA to AVX2 to get CPU independent results
    if device != "cuda":
        env["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
    env["YOLO_VERBOSE"] = "False"  # Set ultralytics to quiet mode

    metrics_file_path = tmp_path / "metrics.json"
    python_executable_with_venv = get_python_executable_with_venv(venv_path)
    run_example_py = EXAMPLE_TEST_ROOT / "run_example.py"
    run_cmd_line = f"{python_executable_with_venv} {run_example_py} --name {example_name} --output {metrics_file_path}"
    if data is not None:
        run_cmd_line += f" --data {data}"

    retry_count = 0
    while True:
        cmd = Command(run_cmd_line, cwd=PROJECT_ROOT, env=env)
        try:
            ret = cmd.run()
            if ret == 0:
                break
        except Exception as e:
            if retry_count >= NUM_RETRY_ON_CONNECTION_ERROR or not _is_connection_error(str(e)):
                raise e
        retry_count += 1
        print(f"Retry {retry_count} after {RETRY_TIMEOUT} seconds")
        time.sleep(RETRY_TIMEOUT)

    measured_metrics = load_json(metrics_file_path)
    print(measured_metrics)
    for name, value in example_params[ACCURACY_METRICS].items():
        assert measured_metrics[name] == pytest.approx(
            value, abs=example_params.get("accuracy_tolerance", ACCURACY_TOLERANCE)
        ), f"metric {name}: {measured_metrics[name]} != {value}"

    if ACCURACY_METRICS_AFTER_TRAINING in example_params:
        for name, value in example_params[ACCURACY_METRICS_AFTER_TRAINING].items():
            assert measured_metrics[name] == pytest.approx(
                value, abs=example_params.get("accuracy_tolerance_after_training", ACCURACY_TOLERANCE)
            ), f"metric {name}: {measured_metrics[name]} != {value}"

    if MODEL_SIZE_METRICS in example_params:
        for name, value in example_params[MODEL_SIZE_METRICS].items():
            assert measured_metrics[name] == pytest.approx(value, rel=MODEL_SIZE_RELATIVE_TOLERANCE)

    if is_check_performance and PERFORMANCE_METRICS in example_params:
        for name, value in example_params[PERFORMANCE_METRICS].items():
            assert measured_metrics[name] == pytest.approx(value, rel=PERFORMANCE_RELATIVE_TOLERANCE)
