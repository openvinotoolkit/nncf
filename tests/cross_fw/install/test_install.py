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

import subprocess
import pytest
from typing import List
from pathlib import Path
import shutil

from tests.shared.paths import PROJECT_ROOT
from tests.shared.paths import TEST_ROOT
from tests.cross_fw.install.conftest import TESTED_BACKENDS
from tests.shared.helpers import create_venv_with_nncf

def run_install_checks(venv_path: Path, tmp_path: Path, package_type: str, backend: str, install_type: str):
    if install_type.lower() not in ['cpu', 'gpu']:
        raise RuntimeError("Unknown installation mode - must be either 'cpu' or 'gpu'")

    python_executable_with_venv = f'. {venv_path}/bin/activate && {venv_path}/bin/python'
    pip_with_venv = f'. {venv_path}/bin/activate && {venv_path}/bin/pip'

    # Do additional install step for sdist/bdist packages
    if package_type == 'sdist':
        for file_path in (PROJECT_ROOT / 'dist').iterdir():
            file_name = str(file_path)
            if file_name.endswith('.tar.gz'):
                package_name = file_name
                break
        else:
            raise FileNotFoundError('NNCF package not found')

        subprocess.run(
            f'{pip_with_venv} install {PROJECT_ROOT}/dist/{package_name}[{backend}] ',
            check=True, shell=True)
    elif package_type == "bdist_wheel":
        subprocess.run(
            f"{pip_with_venv} install {PROJECT_ROOT}/dist/*.whl ", check=True, shell=True)

    run_path = tmp_path / 'run'
    install_checks_py_name = f'install_checks_{backend}.py'
    install_checks_py_path = TEST_ROOT / 'cross_fw' / 'install' / install_checks_py_name
    final_install_checks_py_path = run_path / install_checks_py_name
    shutil.copy(install_checks_py_path, final_install_checks_py_path)
    subprocess.run(
        f'{python_executable_with_venv} {final_install_checks_py_path} {install_type} {package_type}',
        check=True, shell=True, cwd=run_path)


@pytest.fixture(name="venv_type",
                params=["virtualenv", "venv"])
def venv_type_(request):
    return request.param


@pytest.fixture(name="package_type",
                params=["install", "develop", "sdist", "bdist_wheel",
                        "pip_pypi", "pip_local", "pip_e_local", "pip_git_develop"])
def package_type_(request):
    return request.param


@pytest.fixture(name="backend_to_test")
def backend_to_test_(request, backend_clopt: List[str]):
    backends_to_test = set()
    for be in backend_clopt:
        if be == "all":
            backends_to_test.update(TESTED_BACKENDS)
        else:
            backends_to_test.add(be)
    return request.param

def skip_if_backend_not_selected(backend: str, backends_from_cl: List[str]):
    if "all" not in backends_from_cl and backend not in backends_from_cl:
        pytest.skip("not selected for testing")


@pytest.mark.parametrize('backend', TESTED_BACKENDS)
class TestInstall:
    @staticmethod
    def test_install(tmp_path: Path,
                     backend: str, venv_type: str, package_type: str,
                     backend_clopt: List[str], host_configuration_clopt: str):
        skip_if_backend_not_selected(backend, backend_clopt)
        venv_path = create_venv_with_nncf(tmp_path, package_type, venv_type, extra_reqs={backend})
        run_install_checks(venv_path, tmp_path, package_type, backend=backend,
                install_type=host_configuration_clopt)

    @staticmethod
    def test_install_with_tests_requirements(tmp_path: Path,
                                            backend: str, venv_type: str, package_type: str,
                                            backend_clopt: List[str], host_configuration_clopt: str):
        skip_if_backend_not_selected(backend, backend_clopt)
        venv_path = create_venv_with_nncf(tmp_path, package_type, venv_type, extra_reqs={backend})
        pip_with_venv = f'. {venv_path}/bin/activate && {venv_path}/bin/pip'
        subprocess.call(
            f'{pip_with_venv} install -r {PROJECT_ROOT}/tests/{backend}/requirements.txt',
            shell=True)
        run_install_checks(venv_path, tmp_path, package_type, backend=backend, install_type=host_configuration_clopt)
