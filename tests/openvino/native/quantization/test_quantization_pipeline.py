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

import numpy as np
import openvino.runtime as ov
import pytest

import nncf
from nncf.common.quantization.structs import QuantizationPreset
from nncf.openvino.quantization.quantize_model import quantize_impl
from nncf.parameters import TargetDevice
from nncf.scopes import IgnoredScope
from tests.openvino.native.common import get_dataset_for_test
from tests.openvino.native.models import ConvModel
from tests.openvino.native.models import LinearModel
from tests.openvino.native.models import MatMul2DModel
from tests.openvino.native.models import WeightsModel
from tests.openvino.native.test_model_transformer import get_nodes_by_type

REF_FQ_NODES = [
    (("MatMul", 1), ["Input/fq_output_0"]),
    (("Conv", 1), ["Sub/fq_output_0"]),
    (("MatMul", 1), ["Input/fq_output_0"]),
]


@pytest.mark.parametrize("model_creator_func, ref_nodes", zip([LinearModel, ConvModel, MatMul2DModel], REF_FQ_NODES))
def test_compress_weights(model_creator_func, ref_nodes):
    (quntized_op_name, inp_port), ref_fqs_names = ref_nodes
    model = model_creator_func().ov_model
    dataset = get_dataset_for_test(model)
    quantized_model = quantize_impl(
        model,
        dataset,
        preset=QuantizationPreset.PERFORMANCE,
        target_device=TargetDevice.CPU,
        subset_size=1,
        fast_bias_correction=True,
    )

    fq_nodes = get_nodes_by_type(quantized_model, type_name="FakeQuantize")
    assert len(fq_nodes) == len(ref_fqs_names)
    for fq_node in fq_nodes:
        fq_name = fq_node.get_friendly_name()
        assert fq_name in ref_fqs_names

    for op in quantized_model.get_ops():
        if op.get_friendly_name() == quntized_op_name:
            node = op.input_value(inp_port).get_node()
            while node.get_type_name() != "Constant":
                node = node.input_value(0).get_node()
            assert node.get_element_type() == ov.Type(np.int8)
            break


@pytest.mark.parametrize("model_creator_func, ref_nodes", [[ConvModel, REF_FQ_NODES[1]]])
def test_overflow_fix_applied(model_creator_func, ref_nodes):
    (quntized_op_name, inp_port), ref_fqs_names = ref_nodes
    model = model_creator_func().ov_model
    dataset = get_dataset_for_test(model)
    quantized_model = quantize_impl(
        model,
        dataset,
        preset=QuantizationPreset.PERFORMANCE,
        target_device=TargetDevice.CPU,
        subset_size=1,
        fast_bias_correction=True,
    )

    fq_nodes = get_nodes_by_type(quantized_model, type_name="FakeQuantize")
    assert len(fq_nodes) == len(ref_fqs_names)
    for fq_node in fq_nodes:
        fq_name = fq_node.get_friendly_name()
        assert fq_name in ref_fqs_names

    for op in quantized_model.get_ops():
        if op.get_friendly_name() == quntized_op_name:
            node = op.input_value(inp_port).get_node()
            while node.get_type_name() != "Constant":
                node = node.input_value(0).get_node()
            assert node.get_element_type() == ov.Type(np.int8)
            vector = node.get_vector()
            assert np.min(vector) >= -64
            assert np.max(vector) <= 64


IGNORED_OPTIONS = [IgnoredScope(names=["MatMul"]), IgnoredScope(names=["Conv"], types=["Add"]), IgnoredScope()]


@pytest.mark.parametrize(
    "model_creator_func, ignored_options", zip([LinearModel, ConvModel, MatMul2DModel], IGNORED_OPTIONS)
)
def test_meta_information(model_creator_func, ignored_options):
    def check_parameters(quantized_model, parameters, path):
        for key, value in parameters.items():
            rt_path = path + [key]
            if isinstance(value, TargetDevice):
                value = value.value
            if isinstance(value, IgnoredScope):
                if value == IgnoredScope():
                    check_parameters(quantized_model, {"ignored_scope": []}, path)
                    continue
                check_parameters(quantized_model, value.__dict__, rt_path)
                continue
            if "ignored_scope" in path and (not value or key == "validate"):
                assert quantized_model.has_rt_info(rt_path) is False
            else:
                assert quantized_model.get_rt_info(rt_path) == str(value)

    model = model_creator_func().ov_model
    dataset = get_dataset_for_test(model)
    quantize_parameters = {
        "preset": QuantizationPreset.PERFORMANCE,
        "target_device": TargetDevice.CPU,
        "subset_size": 1,
        "fast_bias_correction": True,
        "ignored_scope": ignored_options,
    }
    quantized_model = quantize_impl(model, dataset, **quantize_parameters)

    base_path = ["nncf", "quantization"]
    assert quantized_model.has_rt_info(base_path)

    check_parameters(quantized_model, quantize_parameters, base_path)


@pytest.mark.parametrize(
    "ignored_options, expected_dump",
    [
        (
            IgnoredScope(names=["conv_weights_0", "conv_weights_1"]),
            {
                "validate": None,
                "types": None,
                "subgraphs": None,
                "patterns": None,
                "names": "['conv_weights_0', 'conv_weights_1']",
            },
        ),
        (
            IgnoredScope(
                subgraphs=[
                    nncf.Subgraph(
                        inputs=[
                            "MatMul_1",
                        ],
                        outputs=["MatMul"],
                    )
                ],
            ),
            {
                "validate": None,
                "types": None,
                "subgraphs": "[{'inputs': ['MatMul_1'], 'outputs': ['MatMul']}]",
                "patterns": None,
                "names": None,
            },
        ),
        (
            IgnoredScope(names=["MatMul"], types=["Add"]),
            {
                "validate": None,
                "types": "['Add']",
                "subgraphs": None,
                "patterns": None,
                "names": "['MatMul']",
            },
        ),
        (IgnoredScope(), {"": "[]"}),
    ],
)
def test_ignored_scope_dump(ignored_options, expected_dump, tmp_path):
    ignored_scope_path = ["nncf", "quantization", "ignored_scope"]

    model = WeightsModel().ov_model
    dataset = get_dataset_for_test(model)
    quantize_parameters = {
        "preset": QuantizationPreset.PERFORMANCE,
        "target_device": TargetDevice.CPU,
        "subset_size": 1,
        "fast_bias_correction": True,
        "ignored_scope": ignored_options,
    }
    quantized_model = quantize_impl(model, dataset, **quantize_parameters)
    ov.save_model(quantized_model, tmp_path / "ov_model.xml")
    core = ov.Core()
    dumped_model = core.read_model(tmp_path / "ov_model.xml")
    for key, value in expected_dump.items():
        rt_path = ignored_scope_path + [key] if key else ignored_scope_path
        if value:
            assert dumped_model.get_rt_info(rt_path) == value
        else:
            assert dumped_model.has_rt_info(rt_path) is False
