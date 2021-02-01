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

import subprocess
import sys
import pytest
import shutil
from tests.conftest import TEST_ROOT, PROJECT_ROOT

INSTALL_CHECKS_FILENAME = 'install_checks.py'


@pytest.fixture(name="package_type", params=["install", "develop", "sdist", "bdist_wheel", "pypi"])
def package_type_(request):
    return request.param


def test_install(install_type, tmp_path, package_type):
    if install_type is None:
        pytest.skip("Please specify type of installation")
    venv_path = tmp_path / 'venv'
    venv_path.mkdir()

    version_string = "{}.{}".format(sys.version_info[0], sys.version_info[1])
    subprocess.call("virtualenv -ppython{} {}".format(version_string, venv_path), shell=True)
    python_executable_with_venv = ". {0}/bin/activate && {0}/bin/python".format(venv_path)
    pip_with_venv = ". {0}/bin/activate && {0}/bin/pip".format(venv_path)

    run_path = tmp_path / 'run'
    run_path.mkdir()

    shutil.copy(TEST_ROOT / INSTALL_CHECKS_FILENAME, run_path)

    if package_type == "pypi":
        subprocess.run(
            "{} install nncf".format(pip_with_venv), check=True, shell=True)
    else:

        subprocess.run(
            "{python} {nncf_repo_root}/setup.py {package_type} {install_flag}".format(
                python=python_executable_with_venv,
                nncf_repo_root=PROJECT_ROOT,
                package_type=package_type,
                install_flag='--cpu-only' if
                install_type == "CPU" else ''),
            check=True,
            shell=True,
            cwd=PROJECT_ROOT)

    # Do additional install step for sdist/bdist packages
    if package_type == "sdist":
        subprocess.run(
            "{} install {}/dist/*.tar.gz ".format(pip_with_venv, PROJECT_ROOT), check=True, shell=True)
    elif package_type == "bdist_wheel":
        subprocess.run(
            "{} install {}/dist/*.whl ".format(pip_with_venv, PROJECT_ROOT), check=True, shell=True)

    subprocess.run(
        "{} {}/install_checks.py {}".format(python_executable_with_venv, run_path,
                                            'cpu' if install_type == "CPU" else 'cuda'),
        check=True, shell=True, cwd=run_path)
