"""
 Copyright (c) 2023 Intel Corporation
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

import logging
import sys
import warnings
from contextlib import contextmanager

NNCF_LOGGER_NAME = 'nncf'

nncf_logger = logging.getLogger(NNCF_LOGGER_NAME)
nncf_logger.propagate = False

stdout_handler = logging.StreamHandler(sys.stdout)
fmt_string = '%(levelname)s:%(name)s:%(message)s'
fmt = logging.Formatter(fmt_string)
fmt_no_newline = logging.Formatter(fmt_string, )
stdout_handler.setFormatter(fmt)
stdout_handler.setLevel(logging.INFO)
nncf_logger.addHandler(stdout_handler)
nncf_logger.setLevel(logging.INFO)


def set_log_level(level: int):
    """
    Sets the log level for the NNCF logging.
    :param level: An integer passed into the underlying `logging.Logger` object,
      i.e. the regular `logging` log levels must be used here such as `logging.WARNING`
      or `logging.DEBUG`.
    """
    nncf_logger.setLevel(level)
    for handler in nncf_logger.handlers:
        handler.setLevel(level)


def disable_logging():
    """
    Disables NNCF logging entirely. `FutureWarning`s are still shown.
    """
    nncf_logger.handlers = []


class DuplicateFilter:
    def __init__(self):
        self.msgs = set()

    def filter(self, rec):
        retval = rec.msg not in self.msgs
        self.msgs.add(rec.msg)
        return retval


NNCFDeprecationWarning = FutureWarning


def warning_deprecated(msg):
    # Note: must use FutureWarning in order not to get suppressed by default
    warnings.warn(msg, NNCFDeprecationWarning, stacklevel=2)


@contextmanager
def extension_is_loading_info_log(extension_name: str):
    nncf_logger.info(f"Compiling and loading torch extension: {extension_name}...")
    yield
    nncf_logger.info(f"Finished loading torch extension: {extension_name}")


def warn_bkc_version_mismatch(backend: str, bkc_version: str, current_version: str):
    nncf_logger.warning(
        f"NNCF provides best results with {backend}=={bkc_version}, "
        f"while current {backend} version is {current_version}. "
        f"If you encounter issues, consider switching to {backend}=={bkc_version}")
