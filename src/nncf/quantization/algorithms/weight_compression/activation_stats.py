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

from functools import reduce
from operator import mul

from nncf.experimental.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.tensor import Tensor
from nncf.tensor import functions as fns


def process_stats(stats: WCTensorStatistic, subset_size: int) -> tuple[Tensor, Tensor]:
    """
    A function for processing activations. Shared between AWQ, Scale Estimation and LoRA Correction algorithms.

    :param stats: An object containing statistics for the layer.
    :param subset_size: The number of samples for AWQ.
    :return: tuple of the following tensors:
        s - maximum channel magnitude across samples [HiddenDim]
        X - average channel magnitude across tokens in the sequence [HiddenDim, min(SampleSize, ~subset_size)]
    """
    X = fns.stack(
        stats.mean_values
    )  # [SampleSize, HiddenDim] for 2-D or [SampleSize, No. of Experts, HiddenDim] for 3-D

    # Move SampleSize to the last axis: [HiddenDim, SampleSize] or [No. of Experts, HiddenDim, SampleSize]
    # General approach: move axis 0 to the end
    axes = list(range(1, len(X.shape))) + [0]
    X_full = fns.transpose(X, axes=axes)

    # The sample dimension is always the last axis after transpose
    sample_axis = -1

    # Prevent high memory and time consumption by sampling
    if X_full.shape[sample_axis] > subset_size:
        # dimension of the activation which are unique is chosen here to order indexes by.
        num_dims = len(stats.shape_values[0])
        dim_sets = [set(shape[i] for shape in stats.shape_values) for i in range(num_dims)]
        varying_dims = [i for i in range(num_dims) if len(dim_sets[i]) > 1]

        lens = [reduce(mul, varying_dims, 1) for _ in stats.shape_values]
        step = X_full.shape[sample_axis] // subset_size
        idxs = [i[0] for i in sorted(enumerate(lens), key=lambda x: -x[1])][::step]
        X = X_full[..., idxs]
    else:
        X = X_full

    # Compute max magnitude along the sample axis (last axis)
    # Result: [HiddenDim] or [No. of Experts, HiddenDim]
    s = fns.max(fns.abs(X_full), axis=sample_axis)
    return s, X
