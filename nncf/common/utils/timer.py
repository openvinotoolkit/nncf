# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from contextlib import contextmanager

from nncf.common.logging import nncf_logger


@contextmanager
def timer():
    """
    Context manager to measure execution time.
    """
    start_time = time.perf_counter()
    yield
    elapsed_time = time.perf_counter() - start_time
    time_string = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    nncf_logger.info(f"Elapsed Time: {time_string}")
