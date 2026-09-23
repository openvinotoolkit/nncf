# Copyright (c) 2026 Intel Corporation
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

from pytest import Config
from pytest import Parser

from tests.cross_fw.shared.logging import nncf_caplog  # noqa: F401


def pytest_addoption(parser: Parser):
    parser.addoption(
        "--regen-ref-data",
        action="store_true",
        default=False,
        help="If specified, the reference files will be regenerated using the current state of the repository.",
    )


def pytest_configure(config: Config) -> None:
    regen_dot = config.getoption("--regen-ref-data", False)
    if regen_dot:
        os.environ["NNCF_TEST_REGEN_DOT"] = "1"
