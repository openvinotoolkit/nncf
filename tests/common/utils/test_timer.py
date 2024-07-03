# Copyright (c) 2024 Intel Corporation
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
import time

from nncf.common.utils.timer import timer


def test_timer(nncf_caplog):
    with timer() as t:
        time.sleep(1)

    t()

    with nncf_caplog.at_level(logging.INFO):
        assert "nncf:timer.py:29 Elapsed Time: 00:00:01" in nncf_caplog.text
