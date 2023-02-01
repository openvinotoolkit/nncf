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

import pytest
import numpy as np
import openvino.runtime as ov

from nncf.parameters import TargetDevice
from nncf.common.quantization.structs import QuantizationPreset
from nncf.experimental.openvino_native.quantization.quantize import quantize_impl
from tests.openvino.native.models import LinearModel
from tests.openvino.native.models import ConvModel
from tests.openvino.native.models import MatMul2DModel
from tests.openvino.native.common import get_dataset_for_test
from tests.openvino.native.test_model_transformer import get_fq_nodes

REF_FQ_NODES = [
    (('MatMul', 1), ['Input/fq_output_0']),
    (('Conv', 1), ['Sub/fq_output_0']),
    (('MatMul', 1), ['Input/fq_output_0']),
]


@pytest.mark.parametrize('model_creator_func, ref_nodes', zip([LinearModel, ConvModel, MatMul2DModel], REF_FQ_NODES))
def test_compress_weights(model_creator_func, ref_nodes):
    (quntized_op_name, inp_port), ref_fqs_names = ref_nodes
    model = model_creator_func().ov_model
    dataset = get_dataset_for_test(model)
    quantized_model = quantize_impl(model, dataset, preset=QuantizationPreset.PERFORMANCE,
                                    target_device=TargetDevice.CPU, subset_size=1, fast_bias_correction=True)

    fq_nodes = get_fq_nodes(quantized_model)
    assert len(fq_nodes) == len(ref_fqs_names)
    for fq_name in fq_nodes:
        assert fq_name in ref_fqs_names

    for op in quantized_model.get_ops():
        if op.get_friendly_name() == quntized_op_name:
            node = op.input_value(inp_port).get_node()
            while node.get_type_name() != 'Constant':
                node = node.input_value(0).get_node()
            assert node.get_element_type() == ov.Type(np.int8)
            break
