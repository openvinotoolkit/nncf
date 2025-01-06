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
import tempfile

import pytest

from nncf.common.logging.logger import nncf_logger
from nncf.common.logging.logger import set_log_file

TEST_CASES = [
    (
        [
            ("Test_Message_0", "INFO"),
            ("Test_Message_1", "WARNING"),
            ("Test_Message_2", "ERROR"),
        ],
        [
            "INFO:nncf:Test_Message_0",
            "WARNING:nncf:Test_Message_1",
            "ERROR:nncf:Test_Message_2",
        ],
    )
]


@pytest.mark.parametrize("messages,expected", TEST_CASES)
def test_set_log_file(messages, expected):
    level_to_fn_map = {
        "INFO": nncf_logger.info,
        "WARNING": nncf_logger.warning,
        "ERROR": nncf_logger.error,
    }

    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as tmp_dir:
        log_file = f"{tmp_dir}/log.txt"
        set_log_file(log_file)

        for message, message_level in messages:
            writer = level_to_fn_map[message_level]
            writer(message)

        with open(log_file, "r", encoding="utf8") as f:
            lines = f.readlines()

        for actual_line, expected_line in zip(lines, expected):
            assert actual_line.rstrip("\n") == expected_line

        handlers_to_remove = []

        for handler in nncf_logger.handlers:
            if isinstance(handler, logging.FileHandler) and str(tmp_dir) in handler.baseFilename:
                handler.close()  # so that the log file is released and temp dir can be deleted
                handlers_to_remove.append(handler)
        for h in handlers_to_remove:
            nncf_logger.removeHandler(h)
