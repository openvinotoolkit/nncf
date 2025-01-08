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
import sys
from typing import Set

NNCF_LOGGER_NAME = "nncf"

nncf_logger = logging.getLogger(NNCF_LOGGER_NAME)
nncf_logger.propagate = False

stdout_handler = logging.StreamHandler(sys.stdout)
fmt_string = "%(levelname)s:%(name)s:%(message)s"
fmt = logging.Formatter(fmt_string)
fmt_no_newline = logging.Formatter(
    fmt_string,
)
stdout_handler.setFormatter(fmt)
stdout_handler.setLevel(logging.INFO)
nncf_logger.addHandler(stdout_handler)
nncf_logger.setLevel(logging.INFO)


def set_log_level(level: int) -> None:
    """
    Sets the log level for the NNCF logging.
    :param level: An integer passed into the underlying `logging.Logger` object,
      i.e. the regular `logging` log levels must be used here such as `logging.WARNING`
      or `logging.DEBUG`.
    """
    nncf_logger.setLevel(level)
    for handler in nncf_logger.handlers:
        handler.setLevel(level)


def set_log_file(filename: str) -> None:
    """
    Sets the log file for the NNCF logging.

    :param filename: Path to the file to save the log.
    """
    file_handler = logging.FileHandler(filename, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    nncf_logger.addHandler(file_handler)


def disable_logging() -> None:
    """
    Disables NNCF logging entirely. `FutureWarning`s are still shown.
    """
    nncf_logger.handlers = []


class DuplicateFilter:
    def __init__(self) -> None:
        self.msgs: Set[str] = set()

    def filter(self, rec: logging.LogRecord) -> bool:
        retval = rec.msg not in self.msgs
        self.msgs.add(rec.msg)
        return retval


NNCFDeprecationWarning = FutureWarning


def warn_bkc_version_mismatch(backend: str, bkc_version: str, current_version: str) -> None:
    nncf_logger.warning(
        f"NNCF provides best results with {backend}{bkc_version}, "
        f"while current {backend} version is {current_version}. "
        f"If you encounter issues, consider switching to {backend}{bkc_version}"
    )
