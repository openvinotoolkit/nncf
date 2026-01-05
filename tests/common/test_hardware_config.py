# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import nncf
from nncf import TargetDevice
from nncf.common.hardware.config import get_hw_setup
from nncf.common.hardware.defines import Granularity
from nncf.common.hardware.defines import OpDesc
from nncf.common.hardware.defines import QConfigSpace
from nncf.common.quantization.structs import QuantizationScheme
from nncf.common.quantization.structs import QuantizerConfig


@pytest.mark.parametrize("target_device", TargetDevice)
def test_get_hw_setup(target_device: TargetDevice):
    hw_setup = get_hw_setup(target_device)
    assert len(hw_setup) > 0
    for x in hw_setup:
        assert isinstance(x, OpDesc)


def test_get_hw_setup_error():
    with pytest.raises(nncf.InternalError, match="Unsupported target device:"):
        get_hw_setup(None)


def test_qconfigspace_get_all_qconfigs():
    space = QConfigSpace(
        bits=8,
        mode=(QuantizationScheme.SYMMETRIC, QuantizationScheme.ASYMMETRIC),
        granularity=(Granularity.PER_TENSOR, Granularity.PER_CHANNEL),
        narrow_range=(True, False),
        signedness_to_force=False,
    )
    qconfigs = space.get_all_qconfigs()
    assert len(qconfigs) == 8
    assert all(isinstance(qc, QuantizerConfig) for qc in qconfigs)

    assert qconfigs[0].num_bits == 8
    assert qconfigs[0].signedness_to_force is False
