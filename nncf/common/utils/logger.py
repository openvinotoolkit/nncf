"""
 Copyright (c) 2021 Intel Corporation
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

NNCF_LOGGER_NAME = 'nncf'

logger = logging.getLogger(NNCF_LOGGER_NAME)
logger.propagate = False

stdout_handler = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
stdout_handler.setFormatter(fmt)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)
logger.setLevel(logging.INFO)


def set_log_level(level):
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def disable_logging():
    logger.handlers = []
