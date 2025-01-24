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
import argparse
import atexit
import logging
import queue
import subprocess
import threading
import time
from enum import Enum
from functools import lru_cache
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import psutil
from tabulate import tabulate

logger = logging.getLogger("memory_monitor")


class MemoryType(Enum):
    RSS = "rss"
    SYSTEM = "system"


class MemoryUnit(Enum):
    B = "B"  # byte
    KiB = "KiB"  # Kibibyte
    MiB = "MiB"  # Mibibyte
    GiB = "GiB"  # Gibibyte
    KB = "KB"  # Kilobyte
    MB = "MB"  # Megabyte
    GB = "GB"  # Gigabyte


@lru_cache
def system_memory_warning():
    # Log once
    logger.warning(
        "Please note that MemoryType.SYSTEM in general is affected by other processes that change RAM availability."
    )


class MemoryMonitor:
    def __init__(
        self,
        interval: Optional[float] = 0.1,
        memory_type: Optional[MemoryType] = MemoryType.RSS,
        memory_unit: Optional[MemoryUnit] = MemoryUnit.MiB,
        include_child_processes: Optional[bool] = None,
    ):
        """
        Memory monitoring utility to measure python process memory footprint. After start() is called, it
        creates a thread which runs in parallel and takes memory measurements every *interval* seconds using the
        specified *memory_type* approach. When stop() is called, the memory measuring thread is stopped. The results
        can be obtained by calling get_data(). Memory logs can be saved by calling save_memory_logs(). There are two
        log files: one with data values in a .txt format and another one in a form of a 2D time-memory plot.

        Memory monitor itself allocates some memory itself, especially during figure saving. It is advised to use it
        for measuring large memory processes.

        :param interval: How frequently to take memory measurements (in seconds).
        :param memory_type: Type of memory to log. Accepts four possible values:
            - MemoryType.RSS: Resident Set Size is the portion of memory occupied by a process that is held in RAM.
              Values are obtained through psutil library. If some data is read using mmap, RSS will report this data
              as allocated, however this is not necessarily the case.
            - MemoryType.SYSTEM: This metric is defined as the difference between total system virtual memory
              and system available memory. Be aware, that this way it is affected by other processes that can change
              RAM availability. It is advised to call get_data(memory_from_zero=True) for this type of memory logging,
              if one is interested in memory footprint for a certain process. This subtracts the starting memory from
              all values.

            RSS and SYSTEM behave differently when mmap is used, e.g. during OV model loading. RSS will report data
            which was read with mmap enabled as allocated, however this is not necessarily the case. SYSTEM does not
            report memory loaded with mmap. So it can be used to analyze "pure" memory usage without contribution of
            mmap pages which are actually free, but are reported as allocated by RSS.
        :param memory_unit: Unit to report memory in.
        :param include_child_processes: For MemoryType.RSS only: whether to include memory of child processes. If not
            provided, child processes are counted.
        """
        self.interval = interval
        self.memory_type = memory_type
        if memory_type == MemoryType.SYSTEM:
            system_memory_warning()
        elif memory_type == MemoryType.RSS:
            if include_child_processes is None:
                include_child_processes = True
        else:
            raise ValueError("Unknown memory type to log")
        self.memory_unit = memory_unit
        self.include_child_processes = include_child_processes

        self._monitoring_thread_should_stop = False
        self._monitoring_in_progress = False

        self._memory_monitor_thread = None
        self._memory_values_queue = None
        self._stop_logging_atexit_fn = None

    def start(self, at_exit_fn: Optional[Callable] = None) -> "MemoryMonitor":
        """
        Start memory monitoring.

        :param at_exit_fn: A callable to execute at program exit. Useful fot providing logs saving routine, e.g.
            ```
                at_exit_fn = lambda: memory_monitor.save_memory_logs(*memory_monitor.get_data(), save_dir)
                memory_monitor.start(at_exit_fn=at_exit_fn)
            ```
        """
        if self._monitoring_in_progress:
            raise Exception("Monitoring already in progress")

        self._memory_values_queue = queue.Queue()
        self._monitoring_thread_should_stop = False

        self._memory_monitor_thread = threading.Thread(target=self._monitor_memory)
        self._memory_monitor_thread.daemon = True
        self._memory_monitor_thread.start()
        if at_exit_fn:
            self._stop_logging_atexit_fn = at_exit_fn
            atexit.register(self._stop_logging_atexit_fn)

        self._monitoring_in_progress = True

        return self

    def stop(self):
        """
        Stop memory monitoring.
        """
        if not self._monitoring_in_progress:
            return
        self._monitoring_thread_should_stop = True
        self._monitoring_in_progress = False
        self._memory_monitor_thread.join()
        if self._stop_logging_atexit_fn is not None:
            atexit.unregister(self._stop_logging_atexit_fn)
            self._stop_logging_atexit_fn = None

    def get_data(self, memory_from_zero: Optional[bool] = False) -> Tuple[List, List]:
        """
        :param memory_from_zero: Whether to normalize memory measurements by subtracting the first value. This way
            the measurements will start with 0. Hence, is not very reliable and may actually result in negative values.
        :returns: A tuple of list where the first element corresponds to measurements timestamps and the second one --
        to memory values.
        """
        memory_usage_data = list(self._memory_values_queue.queue)
        if len(memory_usage_data) == 0:
            return [], []
        time_values, memory_values = tuple(zip(*memory_usage_data))
        time_values = _subtract_first_element(list(time_values))
        if memory_from_zero:
            memory_values = _subtract_first_element(list(memory_values))

        # Convert to target memory unit
        memory_values = list(map(partial(_cast_bytes_to, memory_unit=self.memory_unit), memory_values))

        return time_values, memory_values

    def save_memory_logs(
        self,
        time_values: List[float],
        memory_values: List[float],
        save_dir: Path,
        plot_title: Optional[str] = "",
        filename_suffix: Optional[str] = "",
    ):
        """
        Save memory logs as a text file and a 2D plot.

        :param time_values: Timestamps of the memory measurements.
        :param memory_values: Memory measurements.
        :param save_dir: Directory to save logs into.
        :param plot_title: A title for a plot.
        :param filename_suffix: A string suffix to give to the saved files.
        """
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        filename_label = f"{self.memory_type.value}_memory_usage{filename_suffix}"
        # Save measurements to text file
        log_filepath = save_dir / f"{filename_label}.txt"
        with open(log_filepath, "w") as log_file:
            if len(time_values) == 0:
                log_file.write("No measurements recorded.\nPlease make sure logging duration or interval were enough.")
                return
            for timestamp, memory_usage in zip(time_values, memory_values):
                log_file.write(f"{timestamp} {memory_usage:.3f}\n")

            log_file.writelines(
                [
                    f"Total time: {time_values[-1] - time_values[0]}\n",
                    f"Max memory: {max(memory_values):.3f} ({self.memory_unit.value})",
                ]
            )

        # Save measurements plot
        self.save_memory_plot(log_filepath, plot_title)

    def save_memory_plot(self, log_filepath: Path, plot_title: Optional[str] = "", filename_suffix: Optional[str] = ""):
        """
        Parse pre-saved txt file logs and plot a new figure based on this data. May be useful for re-plotting with
        different title.

        :param log_filepath: A path to a .txt log file.
        :param plot_title: A title to give to a plot.
        :param filename_suffix: A string suffix to give to the saved figure.
        """
        with open(log_filepath, "r") as f:
            lines = f.readlines()
            time_values, memory_values = [], []
            for line in lines[:-2]:
                time_value, memory_value = tuple(map(float, line.split(" ")))
                time_values.append(time_value)
                memory_values.append(memory_value)

        fig = plt.figure(figsize=(10, 6))
        plt.plot(time_values, memory_values)
        plt.xlabel("Time (seconds)")
        plt.ylabel(f"Memory Usage ({self.memory_type.value}, {self.memory_unit.value})")
        plt.title(f"{plot_title} Max_{self.memory_type.value}: {max(memory_values):.2f} {self.memory_unit.value}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(log_filepath).replace(".txt", f"{filename_suffix}.png"))
        plt.close(fig)

    def __enter__(self) -> "MemoryMonitor":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _monitor_memory(self):
        while not self._monitoring_thread_should_stop:
            _last_measurement_time = time.perf_counter()
            if self.memory_type == MemoryType.RSS:
                bytes_used = psutil.Process().memory_info().rss
                if self.include_child_processes:
                    for child_process in psutil.Process().children(recursive=True):
                        bytes_used += psutil.Process(child_process.pid).memory_info().rss
            elif self.memory_type == MemoryType.SYSTEM:
                bytes_used = psutil.virtual_memory().total - psutil.virtual_memory().available
            else:
                raise Exception("Unknown memory type to log")
            if self._monitoring_thread_should_stop:
                break
            self._memory_values_queue.put((time.perf_counter(), bytes_used))
            time.sleep(max(0.0, self.interval - (time.perf_counter() - _last_measurement_time)))


class memory_monitor_context:
    def __init__(
        self,
        interval: Optional[float] = 0.1,
        memory_unit: Optional[MemoryUnit] = MemoryUnit.MiB,
        return_max_value: Optional[bool] = True,
        save_dir: Optional[Path] = None,
    ):
        """
        A memory monitor context manager which monitors both RSS and SYSTEM memory types. After, it stores the
        result for the maximum memory recorded if `return_max_value=True or the whole time-memory sequences. Works
        by subtracting the first memory measurement from all the other ones so that the resulting sequence starts
        from 0. Hence, it can actually return negative memory values.

        After exiting, the result is stored at .memory_data field -- a dict with memory types (RSS or SYSTEM)
        as keys. The values are either a single float number if return_max_value is provided, or a tuple with time
        and memory value lists.

        Additionally, memory logs may be saved by providing save_dir argument.

        :param interval: Interval in seconds to take measurements.
        :param memory_unit: Memory unit.
        :param return_max_value: Whether to return max value for each memory type or full memory sequences.
        :param save_dir: If provided, will save memory logs at this location.
        """

        self.memory_monitors = {}
        for memory_type in [MemoryType.RSS, MemoryType.SYSTEM]:
            self.memory_monitors[memory_type] = MemoryMonitor(
                interval=interval, memory_type=memory_type, memory_unit=memory_unit
            )
        self.return_max_value = return_max_value
        self.save_dir = save_dir

        self.memory_data = {}

    def __enter__(self):
        for mm in self.memory_monitors.values():
            mm.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop addition of new values as soon as possible
        for mm in self.memory_monitors.values():
            mm._monitoring_thread_should_stop = True

        for mt, mm in self.memory_monitors.items():
            mm.stop()
            for fz in [False, True]:
                time_values, memory_values = mm.get_data(memory_from_zero=fz)
                if fz:
                    self.memory_data[mt] = max(memory_values) if self.return_max_value else (time_values, memory_values)

                if self.save_dir:
                    mm.save_memory_logs(
                        time_values,
                        memory_values,
                        save_dir=self.save_dir,
                        filename_suffix="_from-zero" if fz else "",
                    )


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
    parser = argparse.ArgumentParser(
        description="Memory Monitor Tool. Monitors memory for an executable and saves logs at specified location.",
        epilog="Examples:\n"
        "   python memory_monitor.py --log-dir ./allocation_logs python allocate.py\n"
        "   python memory_monitor.py optimum-cli export openvino ...",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--log-dir", type=str, default="memory_logs", help="A directory to save logs at. './memory_logs' by default."
    )
    parser.add_argument("executable", type=str, nargs="+", help="Target executable to monitor memory for.")
    args = parser.parse_args()

    memory_monitors = [
        MemoryMonitor(memory_type=mt, include_child_processes=True).start()
        for mt in (MemoryType.RSS, MemoryType.SYSTEM)
    ]

    with subprocess.Popen(" ".join(args.executable), shell=True) as p:
        p.wait()

        # Stop addition of new values as soon as possible
        for mm in memory_monitors:
            mm._monitoring_thread_should_stop = True

    summary_data = []
    for mm in memory_monitors:
        mm.stop()
        for fz in (True, False):
            time_values, memory_values = mm.get_data(memory_from_zero=fz)
            # Most probably the last value is recorded once the child process has already died
            time_values, memory_values = time_values[:-1], memory_values[:-1]
            mm.save_memory_logs(
                time_values, memory_values, save_dir=Path(args.log_dir), filename_suffix="_from-zero" if fz else ""
            )
            summary_data.append([mm.memory_type.value, fz, f"{int(max(memory_values))} {mm.memory_unit.value}"])
    print("\nMemory summary:")
    print(tabulate(summary_data, headers=["Memory type", "From zero", "Peak value"]))
