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

import pytest
import subprocess

from tests.common.helpers import PROJECT_ROOT
from tests.common.helpers import run_install_checks


@pytest.fixture(name="venv_type",
                params=["virtualenv", "venv"])
def venv_type_(request):
    return request.param


@pytest.fixture(name="package_type",
                params=["install", "develop", "sdist", "bdist_wheel",
                        "pip_pypi", "pip_local", "pip_e_local", "pip_git_develop"])
def package_type_(request):
    return request.param


def test_install(tmp_venv_with_nncf, tmp_path, package_type):
    run_install_checks(tmp_venv_with_nncf, tmp_path, package_type, test_dir='tensorflow')


def test_install_with_examples_and_tests_requirements(tmp_venv_with_nncf, tmp_path, package_type):
    pip_with_venv = '. {0}/bin/activate && {0}/bin/pip'.format(tmp_venv_with_nncf)
    subprocess.call(
        '{} install -r {}/examples/tensorflow/requirements.txt'.format(pip_with_venv, PROJECT_ROOT),
        shell=True)
    subprocess.call(
        '{} install -r {}/tests/tensorflow/requirements.txt'.format(pip_with_venv, PROJECT_ROOT),
        shell=True)
    run_install_checks(tmp_venv_with_nncf, tmp_path, package_type, test_dir='tensorflow')
