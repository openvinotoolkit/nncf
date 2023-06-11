# Copyright (c) 2023 Intel Corporation
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
from torch.quantization import FakeQuantize

import nncf
from nncf.data import Dataset
from nncf.experimental.torch.quantization.quantize_model import quantize_impl
from nncf.parameters import TargetDevice
from nncf.quantization import QuantizationPreset
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import BaseQuantizer
from tests.torch.helpers import LeNet
from tests.torch.helpers import RandomDatasetMock


# pylint: disable=too-many-branches
def check_fq(model: NNCFNetwork, striped: bool):
    if hasattr(model.nncf, "external_quantizers"):
        for key in list(model.nncf.external_quantizers.keys()):
            op = model.nncf.external_quantizers[key]
            if striped:
                assert isinstance(op, FakeQuantize)
            else:
                assert isinstance(op, BaseQuantizer)

    for node in model.nncf.get_original_graph().get_all_nodes():
        if node.node_type in ["nncf_model_input", "nncf_model_output"]:
            continue

        nncf_module = model.nncf.get_containing_module(node.node_name)

        if hasattr(nncf_module, "pre_ops"):
            for key in list(nncf_module.pre_ops.keys()):
                op = nncf_module.get_pre_op(key)
                if striped:
                    assert isinstance(op.op, FakeQuantize)
                else:
                    assert isinstance(op.op, BaseQuantizer)

        if hasattr(nncf_module, "post_ops"):
            for key in list(nncf_module.post_ops.keys()):
                op = nncf_module.get_post_ops(key)
                if striped:
                    assert isinstance(op.op, FakeQuantize)
                else:
                    assert isinstance(op.op, BaseQuantizer)


@pytest.mark.parametrize("strip_type", ("nncf", "torch", "nncf_interfere"))
def test_nncf_strip_api(strip_type):
    model = LeNet()
    input_size = [1, 1, 32, 32]

    def transform_fn(data_item):
        images, _ = data_item
        return images

    dataset = Dataset(RandomDatasetMock(input_size), transform_fn)

    quantized_model = quantize_impl(
        model=model,
        calibration_dataset=dataset,
        preset=QuantizationPreset.MIXED,
        target_device=TargetDevice.CPU,
        subset_size=1,
        fast_bias_correction=True,
    )

    if strip_type == "nncf":
        strip_model = nncf.strip(quantized_model)
    elif strip_type == "torch":
        strip_model = nncf.torch.strip(quantized_model)
    elif strip_type == "nncf_interfere":
        strip_model = quantized_model.nncf.strip()

    check_fq(quantized_model, True if strip_model is None else strip_model)
