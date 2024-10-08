# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple

from nncf.tensor import Tensor
from nncf.tensor import functions as fns


def process_stats(stats: Dict[str, List], subset_size: int) -> Tuple[Tensor, Tensor]:
    """
    A function for processing activations. Shared between AWQ, Scale Estimation and LoRA Correction algorithms.

    :param stats: A dictionary containing two lists of length SampleSize. The first list (key "mean_values") contains
        mean statistics tensors. The second one (key "shapes") contains shapes of the original activations before
        reduction. Each element of the fist list is a tensor of shape [HiddenDim] representing a sample activation
        reduced along batch and sequence length dimensions using a "mean" function.
    :param subset_size: The number of samples for AWQ.
    :return: tuple of the following tensors:
        s - maximum channel magnitude across samples [HiddenDim]
        X - average channel magnitude across tokens in the sequence [HiddenDim, min(SampleSize, ~subset_size)]
    """
    X = fns.stack(stats["mean_values"])  # [SampleSize, HiddenDim]
    X_full = fns.transpose(X)  # [HiddenDim, SampleSize]

    # prevent high memory and time consumption
    if X_full.shape[1] > subset_size:
        lens = [shape[0] * shape[1] for shape in stats["shapes"]]
        step = X_full.shape[1] // subset_size
        idxs = [i[0] for i in sorted(enumerate(lens), key=lambda x: -x[1])][::step]
        X = X_full[:, idxs]  # [HiddenDim, ~SubsetSize]
    else:
        X = X_full
    s = fns.max(fns.abs(X_full), axis=1)  # [HiddenDim]
    return s, X
