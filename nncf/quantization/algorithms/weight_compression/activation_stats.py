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

from typing import Tuple

from nncf.tensor import Tensor
from nncf.tensor import functions as fns


def process_stats(stats, subset_size: int) -> Tuple[Tensor, Tensor]:
    """
    It's a processing of activations shared between AWQ, Scale Estimation and LoRA Correction algorithms.

    :param stats: list of activation statistics for a layer that contains N tensors with shape [SeqLen, HiddenDim]
    :type stats: List[TTensor]
    :param subset_size: The number of samples for AWQ.
    :type subset_size: int
    :return: tuple of the following tensors:
        s - maximum channel magnitude across samples [HiddenDim]
        X - average channel magnitude across tokens in the sequence [HiddenDim, SampleSize]
    :rtype: Tuple[TTensor, TTensor]
    """
    X = fns.stack(stats["mean_values"])  # [Batch, HiddenDim]
    X_full = fns.transpose(X)  # [HiddenDim, Batch]

    # prevent high memory and time consumption
    if X_full.shape[1] > subset_size:
        lens = [shape[0] * shape[1] for shape in stats["shapes"]]
        step = X_full.shape[1] // subset_size
        idxs = [i[0] for i in sorted(enumerate(lens), key=lambda x: -x[1])][::step]
        X = X_full[:, idxs]  # [HiddenDim, SampleSize]
    else:
        X = X_full
    s = fns.max(fns.abs(X_full), axis=1)  # [HiddenDim]
    return s, X
