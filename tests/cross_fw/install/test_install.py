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
import shutil
import subprocess
from pathlib import Path
from typing import List

import pytest

from tests.cross_fw.install.conftest import TESTED_BACKENDS
from tests.cross_fw.shared.case_collection import skip_if_backend_not_selected
from tests.cross_fw.shared.helpers import create_venv_with_nncf
from tests.cross_fw.shared.helpers import get_pip_executable_with_venv
from tests.cross_fw.shared.helpers import get_python_executable_with_venv
from tests.cross_fw.shared.paths import PROJECT_ROOT
from tests.cross_fw.shared.paths import TEST_ROOT


def run_install_checks(venv_path: Path, tmp_path: Path, package_type: str, backend: str, install_type: str):
    if install_type.lower() not in ["cpu", "gpu"]:
        raise ValueError("Unknown installation mode - must be either 'cpu' or 'gpu'")

    python_executable_with_venv = get_python_executable_with_venv(venv_path)

    run_path = tmp_path / "run"
    install_checks_py_name = f"install_checks_{backend}.py"
    install_checks_py_path = TEST_ROOT / "cross_fw" / "install" / install_checks_py_name
    final_install_checks_py_path = run_path / install_checks_py_name
    shutil.copy(install_checks_py_path, final_install_checks_py_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)  # need this to be able to import from tests.* in install_checks_*.py
    subprocess.run(
        f"{python_executable_with_venv} {final_install_checks_py_path} {install_type} {package_type}",
        check=True,
        shell=True,
        cwd=run_path,
        env=env,
    )


@pytest.fixture(name="package_type", params=["pip_local", "pip_git_develop", "pip_pypi", "build_s", "build_w"])
def package_type_(request):
    return request.param


@pytest.fixture
def removable_tmp_path(tmp_path: Path):
    # The default tmp_path is automatically removed after some time,
    # but we need to remove the venv after each test to avoid exceeding the space limit.
    yield tmp_path
    shutil.rmtree(tmp_path)


@pytest.fixture(name="backend_to_test")
def backend_to_test_(request, backend_clopt: List[str]):
    backends_to_test = set()
    for be in backend_clopt:
        if be == "all":
            backends_to_test.update(TESTED_BACKENDS)
        else:
            backends_to_test.add(be)
    return request.param


@pytest.mark.parametrize("backend", TESTED_BACKENDS)
class TestInstall:
    @staticmethod
    def test_install(
        removable_tmp_path: Path,
        backend: str,
        package_type: str,
        backend_clopt: List[str],
        host_configuration_clopt: str,
        ov_version_override: str,
    ):
        skip_if_backend_not_selected(backend, backend_clopt)
        if "pypi" in package_type:
            pytest.xfail("Disabled until NNCF is exposed in a release")
        venv_path = create_venv_with_nncf(removable_tmp_path, package_type, "venv", {backend})
        if ov_version_override is not None:
            pip_with_venv = get_pip_executable_with_venv(venv_path)
            ov_version_cmd_line = f"{pip_with_venv} install {ov_version_override}"
            subprocess.run(ov_version_cmd_line, check=True, shell=True)
        run_install_checks(
            venv_path, removable_tmp_path, package_type, backend=backend, install_type=host_configuration_clopt
        )
