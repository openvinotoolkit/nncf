"""
 Copyright (c) 2019 Intel Corporation
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
import sys
from tests.common.helpers import PROJECT_ROOT
try:
    import tensorflow as tf
except ImportError:
    tf = None


@pytest.fixture(scope="function", autouse=True)
def clear_session():
    yield
    if tf:
        tf.keras.backend.clear_session()


def pytest_addoption(parser):
    parser.addoption(
        "--data", type=str, default=None,
        help="Path to test datasets"
    )
    parser.addoption(
        "--sota-checkpoints-dir", type=str, default=None, help="Path to checkpoints directory for sota accuracy test"
    )
    parser.addoption(
        "--sota-data-dir", type=str, default=None, help="Path to datasets directory for sota accuracy test"
    )
    parser.addoption(
        "--metrics-dump-path", type=str, default=None, help="Path to directory to store metrics. "
                                                            "Directory must be empty or should not exist."
                                                            "Metric keeps in "
                                                            "PROJECT_ROOT/test_results/metrics_dump_timestamp "
                                                            "if param not specified"
    )
    parser.addoption(
        "--ov-data-dir", type=str, default=None, help="Path to datasets directory for OpenVino accuracy test"
    )
    parser.addoption(
        "--run-openvino-eval", action="store_true", default=False, help="To run eval models via OpenVino"
    )
    parser.addoption(
        "--run-weekly-tests", action="store_true", default=False, help="To run weekly tests"
    )
    parser.addoption(
        "--models-dir", type=str, default=None, help="Path to checkpoints directory for weekly tests"
    )


@pytest.fixture(scope="module")
def sota_checkpoints_dir(request):
    return request.config.getoption("--sota-checkpoints-dir")


@pytest.fixture(scope="module")
def sota_data_dir(request):
    return request.config.getoption("--sota-data-dir")


@pytest.fixture(scope="module")
def metrics_dump_dir(request):
    pytest.metrics_dump_path = request.config.getoption("--metrics-dump-path")


@pytest.fixture(scope="module")
def dataset_dir(request):
    return request.config.getoption("--data")


@pytest.fixture(scope="module")
def ov_data_dir(request):
    return request.config.getoption("--ov-data-dir")


@pytest.fixture(scope="session")
def openvino(request):
    return request.config.getoption("--run-openvino-eval")


@pytest.fixture(scope="module")
def weekly_tests(request):
    return request.config.getoption("--run-weekly-tests")


@pytest.fixture(scope="module")
def models_dir(request):
    return request.config.getoption("--models-dir")


@pytest.fixture(scope="function")
def tmp_venv_with_nncf(tmp_path, package_type, venv_type):  # pylint:disable=redefined-outer-name
    venv_path = tmp_path / 'venv'
    venv_path.mkdir()

    python_executable_with_venv = ". {0}/bin/activate && {0}/bin/python".format(venv_path)
    pip_with_venv = ". {0}/bin/activate && {0}/bin/pip".format(venv_path)

    version_string = "{}.{}".format(sys.version_info[0], sys.version_info[1])
    if venv_type == 'virtualenv':
        subprocess.call("virtualenv -ppython{} {}".format(version_string, venv_path), shell=True)
    elif venv_type == 'venv':
        subprocess.call("python{} -m venv {}".format(version_string, venv_path), shell=True)
        subprocess.call("{} install --upgrade pip".format(pip_with_venv), shell=True)
        subprocess.call("{} install wheel".format(pip_with_venv), shell=True)

    run_path = tmp_path / 'run'
    run_path.mkdir()

    if package_type == "pip_pypi":
        subprocess.run(
            f"{pip_with_venv} install nncf[tf]", check=True, shell=True)
    elif package_type == "pip_local":
        subprocess.run(
            f"{pip_with_venv} install {PROJECT_ROOT}[tf]", check=True, shell=True)
    elif package_type == "pip_e_local":
        subprocess.run(
            f"{pip_with_venv} install -e {PROJECT_ROOT}[tf]", check=True, shell=True)
    else:

        subprocess.run(
            "{python} {nncf_repo_root}/setup.py {package_type} --tf".format(
                python=python_executable_with_venv,
                nncf_repo_root=PROJECT_ROOT,
                package_type=package_type),
            check=True,
            shell=True,
            cwd=PROJECT_ROOT)

    return venv_path
