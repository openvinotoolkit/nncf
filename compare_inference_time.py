import gc
import time
from unittest.mock import patch

import numpy as np
from tqdm import tqdm

import nncf.utils
from nncf import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.openvino_modeling import OV_MODEL_CACHE
from nncf.quantization.algorithms.weight_compression.openvino_modeling import OVModelParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import calculate_quantized_dequantized_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_int_quantization
from nncf.tensor import Tensor


def get_random_weights(size, amount, n_unique_shapes, dtype, is_sorted=True):
    n_channels = set()
    while len(n_channels) < n_unique_shapes:
        n_channels.add(int(np.random.normal(np.sqrt(size), n_unique_shapes)))
    n_channels = list(n_channels)

    unique_weights = []
    for d in n_channels:
        shape = (size // d, d)
        unique_weights.append(Tensor(np.random.random(shape).astype(dtype)))

    result = []
    for _ in range(amount):
        result.append(np.random.choice(unique_weights))

    if is_sorted:
        result = sorted(result, key=lambda x: x.shape[0] * x.shape[1], reverse=True)
    return result


def measure_compression_time(weights, config, is_ov, verbose=True):
    orig_value = nncf.utils._openvino_available
    nncf.utils._openvino_available = is_ov

    start_time = time.perf_counter()
    for w in tqdm(weights, disable=not verbose):
        do_int_quantization(
            # calculate_quantized_dequantized_weight(
            w,
            config,
            reduction_axes=(1,),
            ov_model_params=OVModelParameters(
                input_dtype=w.dtype,
                output_dtype=None,
                dynamic_shapes=bool(0),
                recompile=bool(0),
                release_memory=bool(1),
                share_inputs=bool(1),
                share_outputs=bool(1),
                return_ov_tensors=bool(0),
            ),
            # return_compressed_weight=bool(1)
        )
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time = total_time / len(weights)
    if verbose:
        print("OV" if is_ov else "NP", f"avg. time: {avg_time:.1e} sec.")

    nncf.utils._openvino_available = orig_value
    OV_MODEL_CACHE.clear()
    gc.collect()
    return avg_time


def bin_search(l, r, config, n, dtype):
    while r / l > 1.05:
        m = np.sqrt(l * r)
        weights = get_random_weights(
            size=int(m),
            amount=n,
            # n_unique_shapes=int(np.sqrt(n)),
            n_unique_shapes=1,
            dtype=dtype,
        )
        t_np = measure_compression_time(
            weights,
            config,
            is_ov=False,
            verbose=False,
        )
        t_ov = measure_compression_time(
            weights,
            config,
            is_ov=True,
            verbose=False,
        )
        print(f"S: {m:.1e}. NP time: {t_np:.1e} sec. OV time: {t_ov:.1e} sec.")
        if t_np < t_ov:
            l = m
        else:
            r = m


N = int(1e5)
S = int(5e5)  # 5e5 for compression/decompression,
K = int(np.sqrt(N))
DTYPE = np.float32

bin_search(
    l=int(1e2),
    r=int(1e5),
    config=WeightCompressionConfig(CompressWeightsMode.INT4_ASYM, group_size=-1),
    n=N,
    dtype=DTYPE,
)

# weights = get_random_weights(size=S, amount=N, n_unique_shapes=K, dtype=np.float32)
# for is_ov in [False, True]:
#     measure_compression_time(
#         weights,
#         WeightCompressionConfig(
#             CompressWeightsMode.INT4_ASYM,
#             group_size=-1
#         ),
#         is_ov=is_ov,
#     )
