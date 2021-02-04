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
from pathlib import Path

import pytest

from nncf.common.utils.logger import logger as nncf_logger

TEST_ROOT = Path(__file__).parent.absolute()
PROJECT_ROOT = TEST_ROOT.parent.absolute()
EXAMPLES_DIR = PROJECT_ROOT / 'examples'


def pytest_addoption(parser):
    parser.addoption(
        "--data", type=str, default=None,
        help="Path to test datasets, e.g. CIFAR10 - for sanity tests or CIFAR100 - for weekly ones"
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


@pytest.yield_fixture()
def _nncf_caplog(caplog):
    nncf_logger.propagate = True
    yield caplog
    nncf_logger.propagate = False


@pytest.fixture(scope="module")
def onnx_dir(request):
    return request.config.getoption("--onnx-dir")


@pytest.fixture(scope="module")
def ov_config_dir(request):
    return request.config.getoption("--ov-config-dir")
