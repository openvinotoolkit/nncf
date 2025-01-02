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

import sys
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.layers import get_per_channel_scale_shape
from nncf.torch.quantization.reference import ReferenceBackendType
from nncf.torch.quantization.reference import ReferenceQuantize
from tools.benchmark import run_profile
from tools.benchmark import run_wall
from tools.benchmark import run_worker

TIME_SCALES = {"ms": 1000}
NBITS = 8
GPU_RUNS_LOW_BATCH = 10000
GPU_RUNS_HIGH_BATCH = 100
CPU_RUNS = 100
LOW_BATCH_INPUT_SIZE = [2, 96, 64, 64]
HIGH_BATCH_INPUT_SIZE = [128, 96, 64, 64]


class BatchMode(Enum):
    LOW = "low"
    HIGH = "high"


class ExecutionType(Enum):
    REGULAR = "regular"
    DATA_PARALLEL = "data_parallel"
    DISTRIBUTED_DATA_PARALLEL = "distributed_data_parallel"


class TimingMode(Enum):
    KERNEL = "kernel"
    WALL = "wall"


@dataclass
class BatchDescriptor:
    mode: BatchMode
    input_size: List[int]
    num_runs: Dict[torch.device, int]


class TensorType(Enum):
    WEIGHTS = "weights"
    ACTIVATIONS = "activations"


class GranularityType(Enum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"


TEST_TENSOR_TYPES: List[TensorType] = [TensorType.WEIGHTS, TensorType.ACTIVATIONS]
TEST_GRANULARITY: List[GranularityType] = [GranularityType.PER_TENSOR, GranularityType.PER_CHANNEL]
TEST_SYMMETRIC: List[bool] = [True, False]
TEST_DEVICES: List[torch.device] = [torch.device("cuda"), torch.device("cpu")]

TEST_BATCHES: List[BatchDescriptor] = [
    BatchDescriptor(
        mode=BatchMode.LOW,
        input_size=LOW_BATCH_INPUT_SIZE,
        num_runs={torch.device("cuda"): GPU_RUNS_LOW_BATCH, torch.device("cpu"): CPU_RUNS},
    ),
    BatchDescriptor(
        mode=BatchMode.HIGH,
        input_size=HIGH_BATCH_INPUT_SIZE,
        num_runs={torch.device("cuda"): GPU_RUNS_HIGH_BATCH, torch.device("cpu"): CPU_RUNS},
    ),
]
TEST_DTYPES: List[torch.dtype] = [torch.float, torch.half]
TEST_EXEC_TYPES: List[ExecutionType] = [
    ExecutionType.REGULAR,
    ExecutionType.DISTRIBUTED_DATA_PARALLEL,
    ExecutionType.DATA_PARALLEL,
]
TEST_NARROW_RANGE: List[bool] = [False, True]
TEST_TIMING_MODE: List[TimingMode] = [TimingMode.WALL, TimingMode.KERNEL]
TEST_REFERENCE: List[bool] = [False, True]


@dataclass
class ParamStruct:
    dtype: torch.dtype
    device: torch.device
    exec_type: ExecutionType
    batch: BatchDescriptor
    tensor_type: TensorType
    granularity: GranularityType
    symmetric: bool
    narrow_range: bool
    timing_mode: TimingMode
    ref: bool

    def to_dict(self) -> Dict:
        dct = asdict(self)
        dct.pop("batch")
        dct["num_runs"] = self.batch.num_runs[self.device]
        dct["input_size"] = self.batch.input_size
        return dct


TEST_PARAM_STRUCTS: List[ParamStruct] = [
    ParamStruct(
        dtype=dtype,
        device=device,
        exec_type=exec_type,
        batch=batch,
        tensor_type=tensor_type,
        granularity=granularity,
        symmetric=symmetric,
        narrow_range=narrow_range,
        timing_mode=timing,
        ref=ref,
    )
    for ref, timing, narrow_range, dtype, exec_type, batch, device, tensor_type, granularity, symmetric, in product(
        TEST_REFERENCE,
        TEST_TIMING_MODE,
        TEST_NARROW_RANGE,
        TEST_DTYPES,
        TEST_EXEC_TYPES,
        TEST_BATCHES,
        TEST_DEVICES,
        TEST_TENSOR_TYPES,
        TEST_GRANULARITY,
        TEST_SYMMETRIC,
    )
    if not (device == torch.device("cpu") and dtype == torch.half)
    and not (device == torch.device("cpu") and exec_type == ExecutionType.DISTRIBUTED_DATA_PARALLEL)
]


class DefaultedPTQuantizerSpec(PTQuantizerSpec):
    def __init__(
        self,
        scale_shape: Tuple[int, ...],
        num_bits: int = 8,
        mode: QuantizationMode = QuantizationMode.SYMMETRIC,
        signedness_to_force: Optional[bool] = None,
        narrow_range: bool = False,
        half_range: bool = False,
        logarithm_scale: bool = None,
    ):
        super().__init__(num_bits, mode, signedness_to_force, narrow_range, half_range, scale_shape, logarithm_scale)


RQ = ReferenceQuantize(backend_type=ReferenceBackendType.TORCH)


def get_module(params_struct: ParamStruct) -> BaseQuantizer:
    input_shape = params_struct.batch.input_size
    is_weights = params_struct.tensor_type == TensorType.WEIGHTS

    scale_shape = [
        1,
    ]
    if params_struct.granularity == GranularityType.PER_CHANNEL:
        scale_shape = get_per_channel_scale_shape(input_shape, is_weights=is_weights)
    specs = DefaultedPTQuantizerSpec(scale_shape=scale_shape, narrow_range=params_struct.narrow_range, num_bits=NBITS)

    module_cls = SymmetricQuantizer if params_struct.symmetric else AsymmetricQuantizer
    m = module_cls(specs)
    m = m.to(params_struct.device)
    if params_struct.dtype == torch.half:
        m.half()

    return m


if __name__ == "__main__":
    file_name = "benchmark_quantize_layers_result.csv" if len(sys.argv) == 1 else sys.argv[1]
    print(f"Benchmark results will be saved to file {file_name}")

    benchmark_data: List[Dict[str, Any]] = []
    device_ids = range(torch.cuda.device_count())
    ngpus_per_node = len(device_ids)
    world_size = ngpus_per_node
    for param_struct in tqdm(TEST_PARAM_STRUCTS):
        param_struct: ParamStruct
        print(param_struct)
        module = get_module(param_struct)
        call_fn = run_wall if param_struct.timing_mode == TimingMode.WALL else run_profile
        num_runs = param_struct.batch.num_runs[param_struct.device]

        input_size = param_struct.batch.input_size
        if param_struct.exec_type == ExecutionType.DISTRIBUTED_DATA_PARALLEL:
            output: List[Dict[str, float]] = []
            try:
                mp.spawn(
                    run_worker,
                    nprocs=ngpus_per_node,
                    args=(world_size, module, input_size, num_runs, param_struct.dtype, output),
                )
                run_data = output[0]
            except:  # noqa: E722
                run_data = {"time": -1}
        else:
            run_data = call_fn(module, input_size, param_struct.device, num_runs, dtype=param_struct.dtype)

        runtime = next(iter(run_data.values()))
        benchmark_data.append({**param_struct.to_dict(), "time_ms": runtime})

        df = pd.DataFrame(benchmark_data)

        df.to_csv(file_name, index=False)
    print("Done!")
