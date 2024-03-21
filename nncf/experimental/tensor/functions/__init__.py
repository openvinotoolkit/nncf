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

from nncf.experimental.tensor.functions import linalg as linalg
from nncf.experimental.tensor.functions.numeric import abs as abs
from nncf.experimental.tensor.functions.numeric import all as all
from nncf.experimental.tensor.functions.numeric import allclose as allclose
from nncf.experimental.tensor.functions.numeric import any as any
from nncf.experimental.tensor.functions.numeric import argsort as argsort
from nncf.experimental.tensor.functions.numeric import as_tensor_like as as_tensor_like
from nncf.experimental.tensor.functions.numeric import astype as astype
from nncf.experimental.tensor.functions.numeric import clip as clip
from nncf.experimental.tensor.functions.numeric import count_nonzero as count_nonzero
from nncf.experimental.tensor.functions.numeric import device as device
from nncf.experimental.tensor.functions.numeric import diag as diag
from nncf.experimental.tensor.functions.numeric import dtype as dtype
from nncf.experimental.tensor.functions.numeric import finfo as finfo
from nncf.experimental.tensor.functions.numeric import flatten as flatten
from nncf.experimental.tensor.functions.numeric import isclose as isclose
from nncf.experimental.tensor.functions.numeric import isempty as isempty
from nncf.experimental.tensor.functions.numeric import item as item
from nncf.experimental.tensor.functions.numeric import matmul as matmul
from nncf.experimental.tensor.functions.numeric import max as max
from nncf.experimental.tensor.functions.numeric import maximum as maximum
from nncf.experimental.tensor.functions.numeric import mean as mean
from nncf.experimental.tensor.functions.numeric import min as min
from nncf.experimental.tensor.functions.numeric import minimum as minimum
from nncf.experimental.tensor.functions.numeric import moveaxis as moveaxis
from nncf.experimental.tensor.functions.numeric import multiply as multiply
from nncf.experimental.tensor.functions.numeric import ones_like as ones_like
from nncf.experimental.tensor.functions.numeric import power as power
from nncf.experimental.tensor.functions.numeric import quantile as quantile
from nncf.experimental.tensor.functions.numeric import reshape as reshape
from nncf.experimental.tensor.functions.numeric import round as round
from nncf.experimental.tensor.functions.numeric import squeeze as squeeze
from nncf.experimental.tensor.functions.numeric import stack as stack
from nncf.experimental.tensor.functions.numeric import sum as sum
from nncf.experimental.tensor.functions.numeric import transpose as transpose
from nncf.experimental.tensor.functions.numeric import unsqueeze as unsqueeze
from nncf.experimental.tensor.functions.numeric import unstack as unstack
from nncf.experimental.tensor.functions.numeric import var as var
from nncf.experimental.tensor.functions.numeric import where as where
from nncf.experimental.tensor.functions.numeric import zeros_like as zeros_like


def _initialize_backends():
    import contextlib

    import nncf.experimental.tensor.functions.numpy_linalg
    import nncf.experimental.tensor.functions.numpy_numeric

    with contextlib.suppress(ImportError):
        import nncf.experimental.tensor.functions.torch_linalg
        import nncf.experimental.tensor.functions.torch_numeric  # noqa: F401


_initialize_backends()
