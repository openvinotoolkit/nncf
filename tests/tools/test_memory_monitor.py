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

import gc
import os
import queue
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from tests.cross_fw.shared.isolation_runner import ISOLATION_RUN_ENV_VAR
from tests.cross_fw.shared.isolation_runner import run_pytest_case_function_in_separate_process
from tools.memory_monitor import MemoryMonitor
from tools.memory_monitor import MemoryType
from tools.memory_monitor import MemoryUnit
from tools.memory_monitor import memory_monitor_context

BYTES_TO_ALLOCATE_SMALL = 2**20  # 1 MiB
BYTES_TO_ALLOCATE_LARGE = 100 * 2**20  # 100 MiB
PREALLOCATE_DURATION = 0.5
ALLOCATE_DURATION = 0.5
DEALLOCATE_DURATION = 0.5
BASELINE_MEMORY_VAR = "TEST_BASELINE_MEMORY"
TEMP_DIR_VAR = "TEST_TEMP_DIR"


if "win32" in sys.platform:
    pytest.skip("Windows is not supported", allow_module_level=True)


def allocate(n_bytes, sleep_before_deallocation=False, sleep_after_deallocation=False):
    if sleep_before_deallocation:
        time.sleep(PREALLOCATE_DURATION)
    data = np.ones((n_bytes,), dtype=np.uint8)
    time.sleep(ALLOCATE_DURATION)
    del data
    gc.collect()
    if sleep_after_deallocation:
        time.sleep(DEALLOCATE_DURATION)


def test_memory_monitor_api(tmpdir):
    tmpdir = Path(tmpdir)

    memory_monitor = MemoryMonitor().start()
    allocate(BYTES_TO_ALLOCATE_SMALL)
    memory_monitor.stop()
    time_values, memory_values = memory_monitor.get_data()

    filename_suffix1 = "_test1"
    memory_monitor.save_memory_logs(
        time_values, memory_values, tmpdir, plot_title="Test", filename_suffix=filename_suffix1
    )
    saved_files = tuple(tmpdir.glob("*"))
    assert len(saved_files) == 2
    assert any(map(lambda fn: str(fn).endswith(f"{filename_suffix1}.txt"), saved_files))
    assert any(map(lambda fn: str(fn).endswith(f"{filename_suffix1}.png"), saved_files))

    filename_suffix2 = "_test2"
    txt_filepath = next(filter(lambda fn: str(fn).endswith(".txt"), saved_files))
    memory_monitor.save_memory_plot(txt_filepath, plot_title="Test re-plot", filename_suffix=filename_suffix2)
    saved_files = list(tmpdir.glob("*"))
    assert len(saved_files) == 3
    assert any(map(lambda fn: str(fn).endswith(f"{filename_suffix2}.png"), saved_files))


@pytest.mark.parametrize("memory_type", (MemoryType.RSS, MemoryType.SYSTEM))
def test_memory_type(memory_type):
    memory_monitor = MemoryMonitor(memory_type=memory_type).start()
    allocate(BYTES_TO_ALLOCATE_SMALL)
    memory_monitor.stop()
    memory_monitor.get_data()


@pytest.mark.parametrize("memory_unit", MemoryUnit.__members__.values())
def test_memory_unit(memory_unit):
    memory_monitor = MemoryMonitor(memory_unit=memory_unit).start()
    allocate(BYTES_TO_ALLOCATE_SMALL)
    memory_monitor.stop()
    memory_monitor.get_data()


@pytest.mark.parametrize("interval", (5e-2, 1e-1))
def test_interval(interval):
    memory_monitor = MemoryMonitor(interval=interval).start()
    allocate(BYTES_TO_ALLOCATE_SMALL, sleep_before_deallocation=True, sleep_after_deallocation=True)
    memory_monitor.stop()
    time_values, memory_values = memory_monitor.get_data()
    assert len(time_values) == len(memory_values)
    assert len(time_values) == pytest.approx(
        (PREALLOCATE_DURATION + ALLOCATE_DURATION + DEALLOCATE_DURATION) / interval, rel=1e-1
    )


@pytest.mark.parametrize("return_max_value", (True, False))
def test_memory_monitor_context(tmpdir, return_max_value):
    tmpdir = Path(tmpdir)
    with memory_monitor_context(return_max_value=return_max_value, save_dir=tmpdir) as mmc:
        allocate(BYTES_TO_ALLOCATE_SMALL)
    memory_data = mmc.memory_data

    assert isinstance(memory_data, dict)
    assert MemoryType.RSS in memory_data
    assert MemoryType.SYSTEM in memory_data
    if return_max_value:
        assert all(map(lambda v: isinstance(v, float), memory_data.values()))
    else:
        assert all(map(lambda v: isinstance(v, tuple) and len(v) == 2 and len(v[0]) == len(v[1]), memory_data.values()))

    saved_files = tuple(tmpdir.glob("*"))
    assert len(saved_files) == 8
    assert sum(map(lambda fn: int(str(fn).endswith(".txt")), saved_files)) == 4
    assert sum(map(lambda fn: int(str(fn).endswith(".png")), saved_files)) == 4


def test_empty_logs(tmpdir):
    memory_monitor = MemoryMonitor().start()
    memory_monitor.stop()
    memory_monitor._memory_values_queue = queue.Queue()  # make sure no logs are recorded
    time_values, memory_values = memory_monitor.get_data()
    assert len(time_values) == len(memory_values) == 0
    memory_monitor.save_memory_logs(time_values, memory_values, Path(tmpdir))


@pytest.mark.skipif(ISOLATION_RUN_ENV_VAR not in os.environ, reason="Should be run via isolation proxy")
def test_memory_values_isolated():
    baseline_memory = float(os.environ[BASELINE_MEMORY_VAR]) if BASELINE_MEMORY_VAR in os.environ else None

    memory_monitor = MemoryMonitor(memory_type=MemoryType.RSS, memory_unit=MemoryUnit.B).start()
    bytes_to_allocate = 1 if baseline_memory is None else BYTES_TO_ALLOCATE_LARGE
    allocate(bytes_to_allocate, sleep_before_deallocation=True, sleep_after_deallocation=True)
    memory_monitor.stop()
    _, memory_values = memory_monitor.get_data()

    if baseline_memory is None:
        print("\nMax memory:", max(memory_values))
    else:
        memory_values = list(map(lambda it: it - baseline_memory, memory_values))
        assert max(memory_values) == pytest.approx(BYTES_TO_ALLOCATE_LARGE, rel=5e-2)
        assert abs(memory_values[0]) < BYTES_TO_ALLOCATE_LARGE * 5e-2
        assert abs(memory_values[-1]) < BYTES_TO_ALLOCATE_LARGE * 5e-2


def test_memory_values():
    # The first run of the test collects the memory that is allocated by default
    _, stdout, _ = run_pytest_case_function_in_separate_process(test_memory_values_isolated)
    max_mem_line = next(filter(lambda s: "Max memory:" in s, stdout.split("\n")))
    baseline_memory = max_mem_line.split(" ")[-1]

    # The second run of the test checks that the added amount of memory is correctly represented by the memory monitor
    os.environ[BASELINE_MEMORY_VAR] = baseline_memory
    run_pytest_case_function_in_separate_process(test_memory_values_isolated)
    del os.environ[BASELINE_MEMORY_VAR]


@pytest.mark.skipif(ISOLATION_RUN_ENV_VAR not in os.environ, reason="Should be run via isolation proxy")
def test_at_exit_isolated():
    memory_monitor = MemoryMonitor()
    at_exit_fn = lambda: memory_monitor.save_memory_logs(*memory_monitor.get_data(), Path(os.environ[TEMP_DIR_VAR]))
    memory_monitor.start(at_exit_fn=at_exit_fn)
    allocate(BYTES_TO_ALLOCATE_SMALL)


def test_at_exit(tmpdir):
    os.environ[TEMP_DIR_VAR] = str(tmpdir)
    run_pytest_case_function_in_separate_process(test_at_exit_isolated)
    del os.environ[TEMP_DIR_VAR]

    tmpdir = Path(tmpdir)
    saved_files = tuple(tmpdir.glob("*"))
    assert len(saved_files) == 2
    assert any(map(lambda fn: str(fn).endswith(".txt"), saved_files))
    assert any(map(lambda fn: str(fn).endswith(".png"), saved_files))
