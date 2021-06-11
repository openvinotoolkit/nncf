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
import os
import subprocess
import sys

from tests.common.helpers import PROJECT_ROOT

import pytest

pytest.register_assert_rewrite('tests.torch.helpers')


def pytest_addoption(parser):
    parser.addoption(
        "--data", type=str, default=None,
        help="Path to test datasets, e.g. CIFAR10 - for sanity tests or CIFAR100 - for weekly ones"
    )

    parser.addoption(
        "--regen-dot", action="store_true", default=False, help="If specified, the "
                                                                "reference .dot files will be regenerated "
                                                                "using the current state of the repository."

    )
    parser.addoption(
        "--torch-home", type=str, default=None, help="Path to cached test models, downloaded by torchvision"
    )
    parser.addoption(
        "--weekly-models", type=str, default=None, help="Path to models' weights for weekly tests"
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
        "--imagenet", action="store_true", default=False, help="Enable tests with imagenet"
    )
    parser.addoption(
        "--test-install-type", type=str, help="Type of installation, use CPU or GPU for appropriate install"
    )
    parser.addoption(
        "--backward-compat-models", type=str, default=None, help="Path to NNCF-traned model checkpoints that are tested"
                                                                 "to be strictly loadable"
    )
    parser.addoption(
        "--third-party-sanity", action="store_true", default=False, help="To run third party sanity test cases"
    )
    parser.addoption(
        "--run-openvino-eval", action="store_true", default=False, help="To run eval models via OpenVino"
    )
    parser.addoption(
        "--onnx-dir", type=str, default=None, help="Path to converted onnx models"
    )
    parser.addoption(
        "--ov-config-dir", type=str, default=None, help="Path to OpenVino configs"
    )
    parser.addoption(
        "--pip-cache-dir", type=str, default=None,
        help="Path to pip cached downloaded packages directory (speeds up installation tests)"
    )



def pytest_configure(config):
    regen_dot = config.getoption('--regen-dot', False)
    if regen_dot:
        os.environ["NNCF_TEST_REGEN_DOT"] = "1"


@pytest.fixture(scope="module")
def dataset_dir(request):
    return request.config.getoption("--data")


@pytest.fixture(scope="module")
def enable_imagenet(request):
    return request.config.getoption("--imagenet")


@pytest.fixture(scope="module")
def weekly_models_path(request):
    return request.config.getoption("--weekly-models")


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
def ov_data_dir(request):
    return request.config.getoption("--ov-data-dir")


@pytest.fixture(scope="module")
def install_type(request):
    return request.config.getoption("--test-install-type")


@pytest.fixture(scope="module")
def backward_compat_models_path(request):
    return request.config.getoption("--backward-compat-models")


@pytest.fixture(autouse=True)
def torch_home_dir(request, monkeypatch):
    torch_home = request.config.getoption("--torch-home")
    if torch_home:
        monkeypatch.setenv('TORCH_HOME', torch_home)


@pytest.fixture(scope="session")
def third_party(request):
    return request.config.getoption("--third-party-sanity")


@pytest.fixture(scope="session")
def openvino(request):
    return request.config.getoption("--run-openvino-eval")


@pytest.fixture(scope="module")
def onnx_dir(request):
    return request.config.getoption("--onnx-dir")


@pytest.fixture(scope="module")
def ov_config_dir(request):
    return request.config.getoption("--ov-config-dir")

@pytest.fixture(scope="module")
def pip_cache_dir(request):
    return request.config.getoption("--pip-cache-dir")


@pytest.fixture(scope="function")
def tmp_venv_with_nncf(install_type, tmp_path, package_type, venv_type):  # pylint:disable=redefined-outer-name
    if install_type is None:
        pytest.skip("Please specify type of installation")
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
            f"{pip_with_venv} install nncf[torch]", check=True, shell=True)
    elif package_type == "pip_local":
        subprocess.run(
            f"{pip_with_venv} install {PROJECT_ROOT}[torch]", check=True, shell=True)
    elif package_type == "pip_e_local":
        subprocess.run(
            f"{pip_with_venv} install -e {PROJECT_ROOT}[torch]", check=True, shell=True)
    else:

        subprocess.run(
            "{python} {nncf_repo_root}/setup.py {package_type} --torch".format(
                python=python_executable_with_venv,
                nncf_repo_root=PROJECT_ROOT,
                package_type=package_type),
            check=True,
            shell=True,
            cwd=PROJECT_ROOT)

    return venv_path


@pytest.fixture
def runs_subprocess_in_precommit():
    # PyTorch caches its CUDA memory allocations, so during the
    # pytest execution the total memory reserved on GPUs will only grow,
    # but it is not necessarily completely occupied at the current moment.
    # The sub-processes are separate to the pytest process and will only see the GPU
    # memory which has not been cached (and thus remains reserved) in the owning pytest process by PyTorch,
    # and the tests below may fail with an OOM. To avoid this, need to call torch.cuda.empty_cache()
    # each time a GPU-powered subprocess is executed during a test.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
