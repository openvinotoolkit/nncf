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
import random
from pathlib import Path

import pytest
from pytest import Config
from pytest import FixtureRequest
from pytest import MonkeyPatch
from pytest import Parser

try:
    import torch
except:  # noqa: E722
    torch = None
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from tests.cross_fw.shared.case_collection import COMMON_SCOPE_MARKS_VS_OPTIONS
from tests.cross_fw.shared.case_collection import skip_marked_cases_if_options_not_specified
from tests.cross_fw.shared.install_fixtures import tmp_venv_with_nncf  # noqa: F401
from tests.cross_fw.shared.logging import nncf_caplog  # noqa: F401

pytest.register_assert_rewrite("tests.torch.helpers")


@pytest.fixture(scope="session", autouse=True)
def disable_tf32_precision():
    if torch:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def pytest_addoption(parser: Parser):
    parser.addoption(
        "--data",
        type=str,
        default=None,
        help="Path to test datasets, e.g. CIFAR10 - for sanity tests or CIFAR100 - for weekly ones",
    )

    parser.addoption(
        "--regen-dot",
        action="store_true",
        default=False,
        help="If specified, the "
        "reference .dot files will be regenerated "
        "using the current state of the repository.",
    )
    parser.addoption(
        "--regen-json",
        action="store_true",
        default=False,
        help="If specified, the "
        "reference .json files will be regenerated "
        "using the current state of the repository.",
    )
    parser.addoption(
        "--torch-home", type=str, default=None, help="Path to cached test models, downloaded by torchvision"
    )
    parser.addoption("--weekly-models", type=str, default=None, help="Path to models' weights for weekly tests")
    parser.addoption(
        "--mixed-precision",
        action="store_true",
        default=False,
        help="Enable mixed precision for the nncf weekly test",
    )
    parser.addoption(
        "--sota-checkpoints-dir", type=Path, default=None, help="Path to checkpoints directory for sota accuracy test"
    )
    parser.addoption(
        "--sota-data-dir", type=Path, default=None, help="Path to datasets directory for sota accuracy test"
    )
    parser.addoption(
        "--metrics-dump-path",
        type=Path,
        default=None,
        help="Path to directory to store metrics. "
        "Directory must be empty or should not exist."
        "Metric keeps in "
        "PROJECT_ROOT/test_results/metrics_dump_timestamp "
        "if param not specified",
    )
    parser.addoption(
        "--ov-data-dir", type=Path, default=None, help="Path to datasets directory for OpenVINO accuracy test"
    )
    parser.addoption("--imagenet", action="store_true", default=False, help="Enable tests with imagenet")
    parser.addoption(
        "--backward-compat-models",
        type=str,
        default=None,
        help="Path to NNCF-traned model checkpoints that are tested to be strictly loadable",
    )
    parser.addoption(
        "--third-party-sanity", action="store_true", default=False, help="To run third party sanity test cases"
    )
    parser.addoption(
        "--torch-with-cuda11",
        action="store_true",
        default=False,
        help="To trigger installation of pytorch with "
        "CUDA11. It's required for 3rd sanity tests "
        "on RTX3090 cards",
    )
    parser.addoption("--run-openvino-eval", action="store_true", default=False, help="To run eval models via OpenVINO")
    parser.addoption("--onnx-dir", type=str, default=None, help="Path to converted onnx models")
    parser.addoption("--ov-config-dir", type=str, default=None, help="Path to OpenVINO configs")
    parser.addoption(
        "--pip-cache-dir",
        type=str,
        default=None,
        help="Path to pip cached downloaded packages directory (speeds up installation tests)",
    )
    parser.addoption(
        "--cuda-ip", type=str, default=None, help="IP address of distributed mode synchronization URL for train test"
    )


def pytest_configure(config: Config):
    for regen_option in ["dot", "json"]:
        if config.getoption(f"--regen-{regen_option}", False):
            os.environ[f"NNCF_TEST_REGEN_{regen_option.upper()}"] = "1"


@pytest.fixture(scope="module")
def dataset_dir(request: FixtureRequest):
    return request.config.getoption("--data")


@pytest.fixture(scope="module")
def enable_imagenet(request: FixtureRequest):
    return request.config.getoption("--imagenet")


@pytest.fixture(scope="module")
def weekly_models_path(request: FixtureRequest):
    return request.config.getoption("--weekly-models")


@pytest.fixture(scope="module")
def mixed_precision(request: FixtureRequest):
    return request.config.getoption("--mixed-precision")


@pytest.fixture(scope="module")
def sota_checkpoints_dir(request: FixtureRequest):
    return request.config.getoption("--sota-checkpoints-dir")


@pytest.fixture(scope="module")
def sota_data_dir(request):
    return request.config.getoption("--sota-data-dir")


@pytest.fixture(scope="module")
def ov_data_dir(request):
    return request.config.getoption("--ov-data-dir")


@pytest.fixture(scope="module")
def install_type(request: FixtureRequest):
    return request.config.getoption("--run-install-tests", skip=True)


@pytest.fixture(scope="module")
def backward_compat_models_path(request: FixtureRequest):
    return request.config.getoption("--backward-compat-models")


@pytest.fixture(autouse=True)
def torch_home_dir(request: FixtureRequest, monkeypatch: MonkeyPatch):
    torch_home = request.config.getoption("--torch-home")
    if torch_home:
        monkeypatch.setenv("TORCH_HOME", torch_home)


@pytest.fixture(scope="session")
def third_party(request: FixtureRequest):
    return request.config.getoption("--third-party-sanity")


@pytest.fixture(scope="session")
def torch_with_cuda11(request: FixtureRequest):
    return request.config.getoption("--torch-with-cuda11")


@pytest.fixture(scope="session")
def openvino(request: FixtureRequest):
    return request.config.getoption("--run-openvino-eval")


@pytest.fixture(scope="module")
def ov_config_dir(request: FixtureRequest):
    return request.config.getoption("--ov-config-dir")


@pytest.fixture(scope="module")
def pip_cache_dir(request: FixtureRequest):
    return request.config.getoption("--pip-cache-dir")


@pytest.fixture(params=[True, False], ids=["per_channel", "per_tensor"])
def is_per_channel(request: FixtureRequest):
    return request.param


@pytest.fixture(params=[True, False], ids=["signed", "unsigned"])
def is_signed(request: FixtureRequest):
    return request.param


@pytest.fixture(params=[True, False], ids=["weights", "activation"])
def is_weights(request: FixtureRequest):
    return request.param


@pytest.fixture(params=[pytest.param(True, marks=pytest.mark.cuda), False], ids=["cuda", "cpu"])
def use_cuda(request: FixtureRequest):
    return request.param


@pytest.fixture(params=[True, False], ids=["half_range", "full_range"])
def is_half_range(request: FixtureRequest):
    return request.param


@pytest.fixture(params=[QuantizationMode.SYMMETRIC, QuantizationMode.ASYMMETRIC])
def quantization_mode(request: FixtureRequest):
    return request.param


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


@pytest.fixture(scope="module")
def distributed_mode_sync_port(request: FixtureRequest):
    return request.config.getoption("--cuda-ip")


@pytest.fixture
def _seed():
    """
    Fixture to ensure deterministic randomness across tests.
    """
    import numpy as np
    from torch.backends import cudnn

    deterministic = cudnn.deterministic
    benchmark = cudnn.benchmark
    seed = torch.seed()

    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    yield

    cudnn.deterministic = deterministic
    cudnn.benchmark = benchmark
    torch.manual_seed(seed)


@pytest.fixture
def _safe_deterministic_state():
    """
    This fixture sets both cuDNN and PyTorch deterministic algorithms to their
    original states after each test to avoid unintended side effects
    caused by modifying these settings globally.
    """
    import torch
    from torch.backends import cudnn

    cudnn_deterministic = cudnn.deterministic
    torch_deterministic = torch.are_deterministic_algorithms_enabled()
    yield
    cudnn.deterministic = cudnn_deterministic
    torch.use_deterministic_algorithms(torch_deterministic)


# Custom markers specifying tests to be run only if a specific option
# is present on the pytest command line must be registered here.
MARKS_VS_OPTIONS = {**COMMON_SCOPE_MARKS_VS_OPTIONS}


def pytest_collection_modifyitems(config, items):
    skip_marked_cases_if_options_not_specified(config, items, MARKS_VS_OPTIONS)
