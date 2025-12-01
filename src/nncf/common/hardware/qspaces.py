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


from nncf.common.hardware.defines import Granularity
from nncf.common.hardware.defines import QConfigSpace
from nncf.common.quantization.structs import QuantizationScheme

q8_a_sym = QConfigSpace(
    bits=8,
    mode=(QuantizationScheme.SYMMETRIC,),
    granularity=(Granularity.PER_TENSOR,),
    narrow_range=(False,),
)

q8_a = QConfigSpace(
    bits=8,
    mode=(QuantizationScheme.SYMMETRIC, QuantizationScheme.ASYMMETRIC),
    granularity=(Granularity.PER_TENSOR,),
    narrow_range=(False,),
)

q8_a_ch = QConfigSpace(
    bits=8,
    mode=(QuantizationScheme.SYMMETRIC, QuantizationScheme.ASYMMETRIC),
    granularity=(Granularity.PER_CHANNEL, Granularity.PER_TENSOR),
    narrow_range=(False,),
)

q8_w_sym = QConfigSpace(
    bits=8,
    mode=(QuantizationScheme.SYMMETRIC,),
    granularity=(Granularity.PER_CHANNEL, Granularity.PER_TENSOR),
    narrow_range=(True,),
    signedness_to_force=True,
)

q8_w_sym_any_nr = QConfigSpace(
    bits=8,
    mode=(QuantizationScheme.SYMMETRIC,),
    granularity=(Granularity.PER_CHANNEL, Granularity.PER_TENSOR),
    narrow_range=(True, False),
    signedness_to_force=True,
)

q8_w_asym = QConfigSpace(
    bits=8,
    mode=(QuantizationScheme.ASYMMETRIC,),
    granularity=(Granularity.PER_CHANNEL, Granularity.PER_TENSOR),
    narrow_range=(False,),
)

q16_a_sym = QConfigSpace(
    bits=16,
    mode=(QuantizationScheme.SYMMETRIC,),
    granularity=(Granularity.PER_TENSOR,),
    narrow_range=(False,),
)

q16_a = QConfigSpace(
    bits=16,
    mode=(QuantizationScheme.SYMMETRIC, QuantizationScheme.ASYMMETRIC),
    granularity=(Granularity.PER_TENSOR,),
    narrow_range=(False,),
)

q16_a_ch = QConfigSpace(
    bits=16,
    mode=(QuantizationScheme.SYMMETRIC, QuantizationScheme.ASYMMETRIC),
    granularity=(Granularity.PER_CHANNEL, Granularity.PER_TENSOR),
    narrow_range=(False,),
)

q16_w_sym = QConfigSpace(
    bits=16,
    mode=(QuantizationScheme.SYMMETRIC,),
    granularity=(Granularity.PER_CHANNEL, Granularity.PER_TENSOR),
    narrow_range=(True,),
    signedness_to_force=True,
)

q16_w_sym_any_nr = QConfigSpace(
    bits=16,
    mode=(QuantizationScheme.SYMMETRIC,),
    granularity=(Granularity.PER_CHANNEL, Granularity.PER_TENSOR),
    narrow_range=(True, False),
    signedness_to_force=True,
)

q16_w_asym = QConfigSpace(
    bits=16,
    mode=(QuantizationScheme.ASYMMETRIC,),
    granularity=(Granularity.PER_CHANNEL, Granularity.PER_TENSOR),
    narrow_range=(False,),
)

q4_tn = QConfigSpace(
    bits=4,
    mode=(QuantizationScheme.SYMMETRIC,),
    granularity=(Granularity.PER_TENSOR,),
    narrow_range=(False,),
)
q4_ch = QConfigSpace(
    bits=4,
    mode=(QuantizationScheme.SYMMETRIC,),
    granularity=(Granularity.PER_CHANNEL,),
    narrow_range=(False,),
)
q4_w = QConfigSpace(
    bits=4,
    mode=(QuantizationScheme.SYMMETRIC,),
    granularity=(Granularity.PER_CHANNEL, Granularity.PER_TENSOR),
    narrow_range=(False,),
)
q2_ch = QConfigSpace(
    bits=2,
    mode=(QuantizationScheme.SYMMETRIC,),
    granularity=Granularity.PER_CHANNEL,
    narrow_range=(False,),
)

q2_w = QConfigSpace(
    bits=2,
    mode=(QuantizationScheme.SYMMETRIC,),
    granularity=(Granularity.PER_CHANNEL, Granularity.PER_TENSOR),
    narrow_range=(False,),
)
