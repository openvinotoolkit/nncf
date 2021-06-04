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

from typing import List, Any

from texttable import Texttable


def create_table(header: List[str], rows: List[List[Any]]) -> str:
    """
    Returns a string which represents a table with a header and rows.

    :param header: Table's header.
    :param rows: Table's rows.
    :return: A string which represents a table with a header and rows.
    """
    return Texttable().header(header).add_rows(rows, header=False).draw()


def is_accuracy_aware_training(config, compression_config_passed=False):
    """
    Returns True if the compression config contains an accuracy-aware
    training related section, False otherwise.
    """
    compression_config = config.get('compression', {}) if not compression_config_passed \
        else config
    if isinstance(compression_config, list):
        for algo_config in compression_config:
            if algo_config.get("accuracy_aware_training") is not None:
                return True
        return False
    if compression_config.get("accuracy_aware_training") is not None:
        return True
    return False
