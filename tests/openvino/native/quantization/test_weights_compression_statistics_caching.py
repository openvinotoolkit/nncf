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
from copy import deepcopy
from functools import partial
from itertools import product

import datasets
import openvino as ov
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.scopes import IgnoredScope


def create_transform_fn(model, tokenizer):
    def transform_fn(data, model=model, tokenizer=tokenizer):
        tokenized_text = tokenizer(data["text"], return_tensors="np")
        input_ids = tokenized_text["input_ids"]

        inputs = {"input_ids": input_ids, "attention_mask": tokenized_text["attention_mask"]}

        batch_size = input_ids.shape[0]
        if hasattr(model, "key_value_input_names"):
            for input_name in model.key_value_input_names:
                model_inputs = model.model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = batch_size
                if shape[2].is_dynamic:
                    shape[2] = 0
                else:
                    shape[1] = 0
                inputs[input_name] = ov.Tensor(model_inputs.get_element_type(), shape.get_shape())

        return inputs

    return transform_fn


def test_weight_compression_statistics_caching_opt_125m(tmp_path, mocker):
    """
    Evaluates the weight compression process for the 'facebook/opt-125m' model,
    specifically focusing on validating the statistics caching mechanism. The test ensures
    that the caching mechanism behaves correctly by:

    1. Collecting statistics once during the compression process.
    2. Loading statistics multiple times, corresponding to the number of different algorithm configurations tested.

    The test iterates over various combinations of compression parameters, including:

    - AWQ
    - Group size for quantization
    - Compression ratios
    - Sensitivity metrics (e.g., Hessian Input Activation)
    - GPTQ
    - Scale estimation
    - Ignored scope
    """
    from nncf.openvino.statistics.aggregator import OVStatisticsAggregator

    collect_statistics_spy = mocker.spy(OVStatisticsAggregator, "collect_statistics")
    load_statistics_from_file_spy = mocker.spy(OVStatisticsAggregator, "load_statistics_from_file")
    dump_statistics_spy = mocker.spy(OVStatisticsAggregator, "dump_statistics")

    # Constant Parameters
    subset_size = 4
    mode = nncf.CompressWeightsMode.INT4_ASYM

    # Compression Parameters
    awq_values = [True, False]
    group_size_values = [1, 4]
    ratio_values = [0.4, 0.8]
    sensitivity_metric_values = [
        nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
        nncf.SensitivityMetric.MAX_ACTIVATION_VARIANCE,
    ]
    gptq_values = [False]  # [True, False] ticket: 155538
    ignored_scope_values = [
        IgnoredScope(),
        IgnoredScope(
            names=[
                "__module.model.model.decoder.layers.1.self_attn.q_proj/prim::PythonOp/MatMul",
                "__module.model.model.decoder.layers.11.self_attn.k_proj/prim::PythonOp/MatMul",
                "__module.model.model.decoder.layers.11.self_attn.k_proj/prim::PythonOp/MatMul",
            ]
        ),
    ]

    MODEL_ID = "facebook/opt-125m"

    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True, load_in_8bit=False, compile=False, stateful=False)

    transform_fn = create_transform_fn(model, tokenizer)
    quantization_dataset = nncf.Dataset(dataset, partial(transform_fn))

    load_statistics_number = 0

    # Test basic configurations (AWQ, group size, ratio, sensitivity metric, ignored_scope)
    for awq, group_size, ratio, sensitivity_metric, ignored_scope in product(
        awq_values, group_size_values, ratio_values, sensitivity_metric_values, ignored_scope_values
    ):
        print(
            f"Testing configuration: awq={awq}, group_size={group_size}, ratio={ratio}, \
            sensitivity_metric={sensitivity_metric}, ignored_scope is empty={len(ignored_scope.names) > 0}"
        )

        # Perform the compression test
        nncf.compress_weights(
            deepcopy(model.model),
            mode=mode,
            dataset=quantization_dataset,
            ratio=ratio,
            awq=awq,
            gptq=False,
            group_size=group_size,
            scale_estimation=False,
            subset_size=subset_size,
            sensitivity_metric=sensitivity_metric,
            ignored_scope=ignored_scope,
            advanced_parameters=AdvancedCompressionParameters(statistics_file_path=tmp_path / "statistics"),
        )
        load_statistics_number += 1

    # Test advanced configurations (GPTQ, scale estimation)
    for gptq, scale_estimation in product(gptq_values, [True, False]):
        print(f"Testing advanced config: awq={True}, gptq={gptq}, scale_estimation={scale_estimation}")

        nncf.compress_weights(
            deepcopy(model.model),
            mode=mode,
            dataset=quantization_dataset,
            ratio=ratio_values[0],
            awq=True,  # AWQ enabled
            gptq=gptq,
            group_size=group_size_values[0],  # Using the first group size
            scale_estimation=scale_estimation,
            subset_size=subset_size,
            sensitivity_metric=sensitivity_metric_values[0],  # Using the first sensitivity metric
            ignored_scope=ignored_scope[0],
            advanced_parameters=AdvancedCompressionParameters(statistics_file_path=tmp_path / "statistics"),
        )
        load_statistics_number += 1

    assert collect_statistics_spy.call_count == 1, "Statistics should be collected only once."
    assert (
        load_statistics_from_file_spy.call_count == load_statistics_number
    ), f"Statistics should be loaded {load_statistics_number} times, \
    but was {load_statistics_from_file_spy.call_count}."
    assert dump_statistics_spy.call_count == 1, "Statistics should be dumped only once."
