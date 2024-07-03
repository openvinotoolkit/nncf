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

from nncf.tensor.functions import linalg as linalg
from nncf.tensor.functions.numeric import abs as abs
from nncf.tensor.functions.numeric import all as all
from nncf.tensor.functions.numeric import allclose as allclose
from nncf.tensor.functions.numeric import any as any
from nncf.tensor.functions.numeric import arange as arange
from nncf.tensor.functions.numeric import argsort as argsort
from nncf.tensor.functions.numeric import as_tensor_like as as_tensor_like
from nncf.tensor.functions.numeric import astype as astype
from nncf.tensor.functions.numeric import ceil as ceil
from nncf.tensor.functions.numeric import clip as clip
from nncf.tensor.functions.numeric import concatenate as concatenate
from nncf.tensor.functions.numeric import count_nonzero as count_nonzero
from nncf.tensor.functions.numeric import device as device
from nncf.tensor.functions.numeric import diag as diag
from nncf.tensor.functions.numeric import dtype as dtype
from nncf.tensor.functions.numeric import expand_dims as expand_dims
from nncf.tensor.functions.numeric import eye as eye
from nncf.tensor.functions.numeric import finfo as finfo
from nncf.tensor.functions.numeric import flatten as flatten
from nncf.tensor.functions.numeric import from_numpy as from_numpy
from nncf.tensor.functions.numeric import isclose as isclose
from nncf.tensor.functions.numeric import isempty as isempty
from nncf.tensor.functions.numeric import item as item
from nncf.tensor.functions.numeric import log2 as log2
from nncf.tensor.functions.numeric import logical_or as logical_or
from nncf.tensor.functions.numeric import masked_mean as masked_mean
from nncf.tensor.functions.numeric import masked_median as masked_median
from nncf.tensor.functions.numeric import matmul as matmul
from nncf.tensor.functions.numeric import max as max
from nncf.tensor.functions.numeric import maximum as maximum
from nncf.tensor.functions.numeric import mean as mean
from nncf.tensor.functions.numeric import median as median
from nncf.tensor.functions.numeric import min as min
from nncf.tensor.functions.numeric import minimum as minimum
from nncf.tensor.functions.numeric import moveaxis as moveaxis
from nncf.tensor.functions.numeric import multiply as multiply
from nncf.tensor.functions.numeric import ones_like as ones_like
from nncf.tensor.functions.numeric import percentile as percentile
from nncf.tensor.functions.numeric import power as power
from nncf.tensor.functions.numeric import quantile as quantile
from nncf.tensor.functions.numeric import reshape as reshape
from nncf.tensor.functions.numeric import round as round
from nncf.tensor.functions.numeric import searchsorted as searchsorted
from nncf.tensor.functions.numeric import squeeze as squeeze
from nncf.tensor.functions.numeric import stack as stack
from nncf.tensor.functions.numeric import sum as sum
from nncf.tensor.functions.numeric import transpose as transpose
from nncf.tensor.functions.numeric import unsqueeze as unsqueeze
from nncf.tensor.functions.numeric import unstack as unstack
from nncf.tensor.functions.numeric import var as var
from nncf.tensor.functions.numeric import where as where
from nncf.tensor.functions.numeric import zeros as zeros
from nncf.tensor.functions.numeric import zeros_like as zeros_like


def _initialize_backends():
    import contextlib

    import nncf.tensor.functions.numpy_linalg
    import nncf.tensor.functions.numpy_numeric

    with contextlib.suppress(ImportError):
        import nncf.tensor.functions.torch_linalg
        import nncf.tensor.functions.torch_numeric  # noqa: F401

    with contextlib.suppress(ImportError):
        import nncf.tensor.functions.ov  # noqa: F401


_initialize_backends()
