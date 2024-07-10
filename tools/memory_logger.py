# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 \\(the "License"\\);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import os
import queue
import subprocess
import threading
import time
from enum import Enum
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import psutil


class MemoryType(Enum):
    RSS = "rss"
    RSS_TOP = "rss_top"
    SYSTEM = "system"
    SYSTEM_NORMALIZED = "system-normalized"


class MemoryUnit(Enum):
    B = "B"  # byte
    KiB = "KiB"  # Kibibyte
    MiB = "MiB"  # Mibibyte
    GiB = "GiB"  # Gibibyte
    KB = "KB"  # Kilobyte
    MB = "MB"  # Megabyte
    GB = "GB"  # Gigabyte


class MemoryLogger:
    def __init__(
        self,
        log_dir: Path,
        plot_title: str = "",
        interval: float = 1,
        memory_type: MemoryType = MemoryType.RSS,
        memory_unit: MemoryUnit = MemoryUnit.MiB,
    ):
        """
        Memory logging utility to measure python process memory footprint. After start_logging() is called, it creates
        a thread which runs in parallel and takes memory measurements every *interval* seconds using the specified
        *memory_type* approach. When stop_logging() is called, the memory measuring thread is stopped and memory report
        is saved into *log_dir* folder. There are two log files: one with data values in a .txt format and another one
        in a form of a 2D time-memory plot. Calling stop_logging() is optional, because it will automatically be called
        at program exit. The example usage is in the end of this file.

        :param log_dir: Directory where to save logged files.
        :param plot_title: Title to put into plot title.
        :param interval: How frequently to take memory measurements (seconds).
        :param memory_type: Type of memory to log. Accepts four possible values:
            - MemoryType.RSS: Resident Set Size is the portion of memory occupied by a process that is held in RAM.
              Values are obtained through psutil library. If some data is read using mmap, RSS will report this data
              as allocated, however this is not necessarily the case.
            - MemoryType.RSS_TOP: The same type of memory as above, but the values are parsed from a table output of
              the *top* command. Usually, reports the same values as RSS, but with worse resolution.
              Note: There can be issues when measuring with RSS_TOP a python script which is run from IDE, in this case
              it should be run from terminal.
            - MemoryType.SYSTEM_NORMALIZED: This metric is defined as the difference between total system virtual memory
              and system available memory. Be aware, that this way it is affected by other processes that can change
              RAM availability. For the same reason it has to be normalized to not take into account memory taken by
              other processed on th system. This is done by subtracting the starting memory from all values, so the
              logging hast to be started int the beginning of the script to not miss any already allocated memory.
            - MemoryType.SYSTEM: Same as above, but not normalized, i.e. will log state of the overall system memory.

            RSS and SYSTEM behave differently when mmap is used, e.g. during OV model loading. RSS will report data
            which was read with mmap enabled as allocated, however this is not necessarily the case. SYSTEM* does not
            report memory loaded with mmap. So it can be used to analyze "pure" memory usage without contribution of
            mmap pages which are actually free, but are reported as allocated by RSS.
        :param memory_unit: Unit to report memory in.
        """
        self.log_dir = log_dir
        self.plot_title = plot_title
        self.interval = interval
        self.memory_type = memory_type
        self.memory_unit = memory_unit

        self._monitoring_thread_should_stop = False
        self._monitoring_in_progress = False

        self._memory_monitor_thread = None
        self._memory_data_queue = None
        self._stop_logging_atexit_fn = None

    def start_logging(self):
        if self._monitoring_in_progress:
            raise Exception("Monitoring already in progress")

        self._memory_data_queue = queue.Queue()
        self._monitoring_thread_should_stop = False

        self._memory_monitor_thread = threading.Thread(target=self._monitor_memory)
        self._memory_monitor_thread.daemon = True
        self._memory_monitor_thread.start()
        self._stop_logging_atexit_fn = lambda: self.stop_logging()
        atexit.register(self._stop_logging_atexit_fn)

        self._monitoring_in_progress = True

        return self

    def stop_logging(self):
        self._monitoring_thread_should_stop = True
        self._monitoring_in_progress = False
        self._memory_monitor_thread.join()
        self._log_memory_usage()
        atexit.unregister(self._stop_logging_atexit_fn)

    def _log_memory_usage(self, filename_suffix=""):
        memory_usage_data = list(self._memory_data_queue.queue)
        time_data, memory_data = tuple(zip(*memory_usage_data))
        time_data = _subtract_first_element(list(time_data))
        if self.memory_type == MemoryType.SYSTEM_NORMALIZED:
            memory_data = _subtract_first_element(list(memory_data))

        # Convert to target memory unit
        memory_data = list(map(partial(_cast_bytes_to, memory_unit=self.memory_unit), memory_data))

        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

        filename_label = f"{self.memory_type.value}_memory_usage{filename_suffix}"
        # Save measurements to file
        with open(self.log_dir / f"{filename_label}.txt", "w") as log_file:
            for timestamp, memory_usage in zip(time_data, memory_data):
                log_file.write(f"{timestamp} {memory_usage:.3f}\n")

            log_file.writelines(
                [
                    f"Total time: {time_data[-1] - time_data[0]}\n",
                    f"Max memory: {max(memory_data):.3f} ({self.memory_unit.value})",
                ]
            )

        # Save memory plot
        fig = plt.figure(figsize=(10, 6))
        plt.plot(time_data, memory_data)
        plt.xlabel("Time (seconds)")
        plt.ylabel(f"Memory Usage ({self.memory_type.value}, {self.memory_unit.value})")
        plt.title(self.plot_title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.log_dir / f"{filename_label}.png")
        plt.close(fig)

    def _monitor_memory(self):
        while not self._monitoring_thread_should_stop:
            if self.memory_type == MemoryType.RSS:
                bytes_used = psutil.Process().memory_info().rss
            elif self.memory_type == MemoryType.RSS_TOP:
                elem_filter = lambda it: "\x1b" not in it
                new_line_delimiter = "\x1b(B\x1b[m\x1b[39;49m"
                header_line = -3
                res_column = 5  # Resident Memory Size (KiB): The non-swapped physical memory a task is using.

                res = subprocess.run(
                    f"top -n 1 -p {os.getpid()}".split(" "),
                    capture_output=True,
                    text=True,
                )
                stdout, _ = res.stdout, res.stderr
                lines = stdout.split(new_line_delimiter)
                if len(lines) < abs(header_line):
                    continue
                assert tuple(filter(elem_filter, lines[header_line].split()))[res_column] == "RES"
                line_elems = tuple(filter(elem_filter, lines[header_line + 1].split()))
                res_data = line_elems[res_column]
                if res_data.endswith("m") or res_data.endswith("g"):
                    float_value = float(res_data[:-1].replace(",", "."))
                    bytes_used = float_value * 2 ** (30 if "g" in res_data else 20)
                else:
                    bytes_used = float(res_data) * 2**10
            elif self.memory_type in [MemoryType.SYSTEM, MemoryType.SYSTEM_NORMALIZED]:
                bytes_used = psutil.virtual_memory().total - psutil.virtual_memory().available
            else:
                raise Exception("Unknown memory type to log")
            self._memory_data_queue.put((time.perf_counter(), bytes_used))
            time.sleep(self.interval)


def _cast_bytes_to(bytes, memory_unit, round_to_int=False):
    memory_unit_divisors = {
        MemoryUnit.B: 1,
        MemoryUnit.KiB: 2**10,
        MemoryUnit.MiB: 2**20,
        MemoryUnit.GiB: 2**30,
        MemoryUnit.KB: 10**3,
        MemoryUnit.MB: 10**6,
        MemoryUnit.GB: 10**9,
    }
    result = bytes / memory_unit_divisors[memory_unit]
    return int(result) if round_to_int else result


def _subtract_first_element(data):
    for i in range(1, len(data)):
        data[i] = data[i] - data[0]
    data[0] = 0
    return data


if __name__ == "__main__":
    memory_loggers = []
    for memory_type in [MemoryType.RSS, MemoryType.SYSTEM_NORMALIZED]:
        memory_loggers.append(MemoryLogger(Path("./logs"), memory_type=memory_type).start_logging())
    import numpy as np
    from tqdm import tqdm

    a = []
    for i in tqdm(range(10)):
        a.append(np.random.random((1 << 20,)))
        time.sleep(2)
    del a
    time.sleep(2)
    # Optional:
    # map(lambda ml: ml.stop_logging(), memory_loggers)
