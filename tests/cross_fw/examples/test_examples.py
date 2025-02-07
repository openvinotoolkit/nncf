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
from pathlib import Path
from typing import Any, Dict, List

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


def example_test_cases():
    example_scope = load_json(EXAMPLE_SCOPE_PATH)
    for example_name, example_params in example_scope.items():
        yield pytest.param(example_name, example_params, id=example_name)


@pytest.mark.parametrize("example_name, example_params", example_test_cases())
def test_examples(
    tmp_path: Path,
    example_name: str,
    example_params: Dict[str, Any],
    backends_list: List[str],
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
    skip_if_backend_not_selected(backend, backends_list)
    if reuse_venv:
        # Use example directory as tmp_path
        tmp_path = Path(example_params["requirements"]).parent
    venv_path = create_venv_with_nncf(tmp_path, "pip_e_local", "venv", {backend})
    pip_with_venv = get_pip_executable_with_venv(venv_path)
    if "requirements" in example_params:
        requirements = PROJECT_ROOT / example_params["requirements"]
        run_cmd_line = f"{pip_with_venv} install -r {requirements}"
        subprocess.run(run_cmd_line, check=True, shell=True)

    if ov_version_override is not None:
        ov_version_cmd_line = f"{pip_with_venv} install {ov_version_override}"
        subprocess.run(ov_version_cmd_line, check=True, shell=True)

    subprocess.run(f"{pip_with_venv} list", check=True, shell=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)  # need this to be able to import from tests.* in run_example.py
    env["ONEDNN_MAX_CPU_ISA"] = "AVX2"  # Set ISA to AVX2 to get CPU independent results
    env["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
    env["YOLO_VERBOSE"] = "False"  # Set ultralytics to quiet mode

    metrics_file_path = tmp_path / "metrics.json"
    python_executable_with_venv = get_python_executable_with_venv(venv_path)
    run_example_py = EXAMPLE_TEST_ROOT / "run_example.py"
    run_cmd_line = f"{python_executable_with_venv} {run_example_py} --name {example_name} --output {metrics_file_path}"
    if data is not None:
        run_cmd_line += f" --data {data}"
    cmd = Command(run_cmd_line, cwd=PROJECT_ROOT, env=env)
    cmd.run()

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
