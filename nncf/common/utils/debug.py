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
from contextlib import contextmanager
from typing import Generator

from nncf.common.logging import nncf_logger

DEBUG_LOG_DIR = "./nncf_debug"


def is_debug() -> bool:
    return nncf_logger.getEffectiveLevel() == logging.DEBUG


def set_debug_log_dir(dir_: str) -> None:
    global DEBUG_LOG_DIR
    DEBUG_LOG_DIR = dir_


@contextmanager
def nncf_debug() -> Generator[None, None, None]:
    from nncf.common.logging.logger import set_log_level

    set_log_level(logging.DEBUG)
    yield
    set_log_level(logging.INFO)
