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
import logging
import os
import random

import pytest
import torch
from pytest import Config
from pytest import FixtureRequest
from pytest import Parser

from nncf import set_log_level

pytest.register_assert_rewrite("tests.torch.helpers")


@pytest.fixture(scope="session", autouse=True)
def disable_tf32_precision():
    if torch:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def pytest_addoption(parser: Parser):
    parser.addoption(
        "--regen-ref-data",
        action="store_true",
        default=False,
        help="If specified, the reference files will be regenerated using the current state of the repository.",
    )
    parser.addoption(
        "--nncf-debug",
        action="store_true",
        default=False,
        help="Set debug level for nncf logger.",
    )


def pytest_configure(config: Config) -> None:
    regen_dot = config.getoption("--regen-ref-data", False)
    if regen_dot:
        os.environ["NNCF_TEST_REGEN_DOT"] = "1"

    nncf_debug = config.getoption("--nncf-debug", False)
    if nncf_debug:
        set_log_level(logging.DEBUG)

    # Disable patching of torch functions
    os.environ["NNCF_EXPERIMENTAL_TORCH_TRACING"] = "1"


@pytest.fixture
def regen_ref_data(request: FixtureRequest):
    return request.config.getoption("--regen-ref-data", False)


@pytest.fixture(params=[pytest.param(True, marks=pytest.mark.cuda), False], ids=["cuda", "cpu"])
def use_cuda(request: FixtureRequest):
    return request.param


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
