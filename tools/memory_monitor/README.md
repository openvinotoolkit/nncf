
# Memory Monitor

Memory Monitor is a tool that can be used to measure python program RAM footprint in time. It supports multiple memory types:

- `MemoryType.RSS`: Resident Set Size is the portion of memory occupied by a process that is held in RAM.

- `MemoryType.SYSTEM`: This metric is defined as the difference between total system virtual memory and system available memory. Be aware, that this way it is affected by other processes that can change RAM availability. It is advised to call `get_data(memory_from_zero=True)` for this type of memory logging, if one is interested in memory footprint for a certain process. This subtracts the starting memory from all values.

RSS and SYSTEM behave differently when mmap is used, e.g. during OV model loading. RSS will report data which was read with mmap enabled as allocated, however this is not necessarily the case. SYSTEM does not report memory loaded with mmap. So it can be used to analyze "pure" memory usage without contribution of mmap pages which most probably will actually be free, but are reported as allocated by RSS.

It is advised to use `MemoryType.SYSTEM` when analyzing memory of python scripts involving OpenVINO model reading. Also, memory monitor itself allocates some memory itself, especially during figure saving. It is advised to use it for measuring large memory processes.

## Example 1. Monitor for an executable

The tool allows to monitor memory for some executable including other python scripts. For example:

```shell
python memory_monitor.py --log-dir ./allocation_logs python allocate.py
```

```shell
python memory_monitor.py "optimum-cli export openvino ..."
```

## Example 2. As a python Module

```python
import gc
import time
import numpy as np

from functools import partial
from pathlib import Path
from tqdm import tqdm

from memory_monitor import MemoryType
from memory_monitor import memory_monitor_context

save_dir = Path("memory_logs")


def allocate_memory():
    a = []
    for _ in tqdm(range(10)):
        a.append(np.random.random((1 << 25,)))
        time.sleep(1)
    return a

# Define a helper logging function
def log(mm, fz):
    mm.save_memory_logs(
        *mm.get_data(memory_from_zero=fz),
        save_dir=save_dir,
        filename_suffix="_from-zero" if fz else ""
    )

# Create three memory monitors with different memory types and logging parameters
memory_monitor_configurations = [
    (MemoryType.RSS, False),
    (MemoryType.SYSTEM, False),
    (MemoryType.SYSTEM, True)
]
for memory_type, mem_from_zero in memory_monitor_configurations:
    memory_monitor = MemoryMonitor(memory_type=memory_type)
    # Start logging and register a logging function that will save logs at exit
    memory_monitor.start(at_exit_fn=partial(log, memory_monitor, mem_from_zero))

# Example logic allocating some memory
a = allocate_memory()
del a
gc.collect()
time.sleep(1)
```

## Example 3. Memory Monitor Context

Alternatively, you may use `memory_monitor_context` that envelops logic for creating MemoryMonitors and saving logs. It can also return only the maximal memory value if needed. Memory data will be available at `context.memory_data`.

```python
import gc
import time
import numpy as np

from pathlib import Path
from tqdm import tqdm

from memory_monitor import MemoryType
from memory_monitor import memory_monitor_context

save_dir = Path("memory_logs")


def allocate_memory():
    a = []
    for _ in tqdm(range(10)):
        a.append(np.random.random((1 << 25,)))
        time.sleep(1)
    return a

with memory_monitor_context(return_max_value=True, save_dir="memory_logs") as mmc:
    a = allocate_memory()
    del a
    gc.collect()
    time.sleep(1)

max_memory_usage: float = mmc.memory_data[MemoryType.SYSTEM]
```
