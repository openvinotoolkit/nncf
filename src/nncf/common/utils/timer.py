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

import time
from contextlib import contextmanager
from typing import Callable, Iterator

from nncf.common.logging import nncf_logger


@contextmanager
def timer() -> Iterator[Callable[[], float]]:
    """
    Context manager to measure execution time.
    """
    start_time = end_time = time.perf_counter()
    yield lambda: end_time - start_time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    time_string = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    nncf_logger.info(f"Elapsed Time: {time_string}")
