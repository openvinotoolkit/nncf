"""
 Copyright (c) 2020 Intel Corporation
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

import os
import subprocess
import pytest
import shutil
from tests.common.helpers import TEST_ROOT, PROJECT_ROOT

INSTALL_CHECKS_FILENAME = 'install_checks.py'


@pytest.fixture(name="venv_type",
                params=["virtualenv", "venv"])
def venv_type_(request):
    return request.param


@pytest.fixture(name="package_type",
                params=["install", "develop", "sdist", "bdist_wheel",
                        "pip_pypi", "pip_local", "pip_e_local"])
def package_type_(request):
    return request.param


def test_install(tmp_venv_with_nncf, install_type, tmp_path, package_type):
    venv_path = tmp_venv_with_nncf

    python_executable_with_venv = ". {0}/bin/activate && {0}/bin/python".format(venv_path)
    pip_with_venv = ". {0}/bin/activate && {0}/bin/pip".format(venv_path)

    run_path = tmp_path / 'run'

    shutil.copy(TEST_ROOT / 'torch' / INSTALL_CHECKS_FILENAME, run_path)

    # Do additional install step for sdist/bdist packages
    if package_type == "sdist":
        for file_name in os.listdir(os.path.join(PROJECT_ROOT, 'dist')):
            if file_name.endswith('.tar.gz'):
                package_name = file_name
                break
        else:
            raise FileNotFoundError('NNCF package not found')
        subprocess.run(
            "{} install {}/dist/{}[torch] ".format(pip_with_venv, PROJECT_ROOT, package_name), check=True, shell=True)
    elif package_type == "bdist_wheel":
        subprocess.run(
            "{} install {}/dist/*.whl ".format(pip_with_venv, PROJECT_ROOT), check=True, shell=True)

    subprocess.run(
        "{} {}/install_checks.py {} {}".format(python_executable_with_venv, run_path,
                                               'cpu' if install_type == "CPU" else 'cuda',
                                               package_type),
        check=True, shell=True, cwd=run_path)
