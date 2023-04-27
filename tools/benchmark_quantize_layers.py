"""
 Copyright (c) 2023 Intel Corporation
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

import os
from itertools import product
from typing import Any, Optional, Tuple

import pandas as pd
import torch
import torch.multiprocessing as mp
from torch import nn
from tqdm import tqdm

from nncf.common.quantization.structs import QuantizationMode
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.layers import get_per_channel_scale_shape
from nncf.torch.utils import sum_like
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


TEST_PLACES = ["weights", "activations"]
TEST_GRANULARITY = ["per_tensor", "per_channel"]
TEST_SYMMETRIC = [True, False]
TEST_DEVICES = [torch.device("cuda"), torch.device("cpu")]
TEST_BATCHES = [
    {
        "mode": "low batch",
        "input_size": LOW_BATCH_INPUT_SIZE,
        "runs": {torch.device("cuda"): GPU_RUNS_LOW_BATCH, torch.device("cpu"): CPU_RUNS},
    },
    {
        "mode": "high batch",
        "input_size": HIGH_BATCH_INPUT_SIZE,
        "runs": {torch.device("cuda"): GPU_RUNS_HIGH_BATCH, torch.device("cpu"): CPU_RUNS},
    },
]
TEST_DTYPES = [torch.float, torch.half]
TEST_DISTR_MODE = ["SYNK", "DATAPARALLEL", "DATADISTRIBUTED"]
TEST_NARROW_RANGE = [False, True]
TEST_TIMING_MODE = ["KERNEL", "WALL"]
TEST_REFERENCE = [True, False]

TEST_PARAMS_STRUCT = [
    {
        "dtype": dtype,
        "device": device,
        "batch": batch,
        "place": place,
        "granularity": granularity,
        "symmetric": symmetric,
        "narrow_range": narrow_range,
        "mode": distr_mode,
        "timing": timing,
        "ref": ref,
    }
    for dtype, device, distr_mode, place, granularity, symmetric, narrow_range, timing, ref, batch in product(
        TEST_DTYPES,
        TEST_DEVICES,
        TEST_DISTR_MODE,
        TEST_PLACES,
        TEST_GRANULARITY,
        TEST_SYMMETRIC,
        TEST_NARROW_RANGE,
        TEST_TIMING_MODE,
        TEST_REFERENCE,
        TEST_BATCHES,
    )
    if not (device == torch.device("cpu") and dtype == torch.half)
    and not (device == torch.device("cpu") and distr_mode != "SYNK")
    and not (device == torch.device("cuda") and distr_mode != "SYNK" and batch["mode"] == "low_batch")
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


# reference impl
class ReferenceQuantizeSymmetric(torch.autograd.Function):
    # pylint:disable=abstract-method
    @staticmethod
    def forward(ctx, input_, scale, bits):
        level_high = scale.new_tensor([2 ** (bits - 1) - 1])
        level_low = scale.new_tensor([-(level_high + 1)])
        s = level_high / scale

        output = input_ * s
        output = output.clamp(min=level_low[0], max=level_high[0])
        output = output.round()
        output = output / s

        ctx.save_for_backward(input_, scale, output)
        ctx.level_high = level_high
        ctx.level_low = level_low

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        input_, scale, output = ctx.saved_tensors
        level_high = ctx.level_high
        level_low = ctx.level_low

        alpha = float(level_low) / float(level_high)
        mask_hi = (input_ > scale).type(input_.dtype)
        mask_lo = (input_ < scale * alpha).type(input_.dtype)
        mask_in = 1 - mask_hi - mask_lo

        val_grad_out = mask_hi + alpha * mask_lo
        err = (output - input_) * scale.reciprocal()
        grad_scale = grad_output * (err * mask_in + val_grad_out)
        grad_scale = sum_like(grad_scale, scale)

        # calc gradient for input
        grad_input = grad_output * mask_in

        return grad_input, grad_scale, None


class ReferenceQuantize(nn.Module):
    def __init__(self, num_bits=8, input_shape=None, is_weights=True, per_channel=False):
        super().__init__()
        self.input_shape = input_shape
        self.is_weights = is_weights
        scale_shape = [1]
        if per_channel:
            scale_shape = get_per_channel_scale_shape(self.input_shape, self.is_weights)

        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.num_bits = num_bits
        self.level_high = 2 ** (self.num_bits - 1) - 1
        self.level_low = -(self.level_high + 1)
        self.quantize = ReferenceQuantizeSymmetric.apply

    def get_scale(self):
        return self.scale

    def forward(self, input_):
        return self.quantize(input_, self.scale, self.num_bits)


def get_module(params, per_tensor_scale_shape):
    input_shape = params["batch"]["input_size"]
    is_weights = params["place"] == "weights"

    if params["ref"]:
        module = ReferenceQuantize(NBITS, input_shape, is_weights, per_channel=params["granularity"] == "per_channel")
    else:
        scale_shape = per_tensor_scale_shape
        if params["granularity"] == "per_channel":
            scale_shape = get_per_channel_scale_shape(input_shape, is_weights=is_weights)
        specs = DefaultedPTQuantizerSpec(scale_shape=scale_shape, narrow_range=params["narrow_range"], num_bits=NBITS)

        module_cls = SymmetricQuantizer if params["symmetric"] else AsymmetricQuantizer
        module = module_cls(specs)

    module = module.to(params["device"])
    if params["dtype"] == torch.half:
        module.half()

    if params["ref"] and params["mode"] == "DATAPARALLEL":
        module = nn.parallel.DataParallel(module, range(torch.cuda.device_count()))
    return module


if __name__ == "__main__":
    file_name = "benchmark_quantize_layers_result.csv" if len(os.argv) == 1 else os.argv[1]
    print(f"Benchmark results will be saved to file {file_name}")

    benchmark_data = []
    per_tensor_scale_shape = (1,)
    device_ids = range(torch.cuda.device_count())
    ngpus_per_node = len(device_ids)
    world_size = ngpus_per_node
    for params in tqdm(TEST_PARAMS_STRUCT):
        print(params)
        module = get_module(params, per_tensor_scale_shape)
        call_fn = run_wall if params["timing"] == "WALL" else run_profile
        runs = params["batch"]["runs"][params["device"]]

        input_size = params["batch"]["input_size"]
        if params["mode"] == "DATADISTRIBUTED":
            mp.spawn(
                run_worker,
                nprocs=ngpus_per_node,
                args=(world_size, module, input_size, runs, params["dtype"], benchmark_data),
            )
        else:
            call_fn(module, input_size, params["device"], runs, dtype=params["dtype"], output=benchmark_data)
        batch_data = params.pop("batch")
        batch_data.update({"runs": batch_data["runs"][params["device"]]})
        params.update(batch_data)
        benchmark_data[-1] = {**params, **benchmark_data[-1]}

    df = pd.DataFrame(benchmark_data)

    df.to_csv(file_name, index=False)
    print("Done!")
