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
from copy import deepcopy
from functools import partial
from itertools import product
from typing import Tuple

import datasets
import openvino as ov
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.scopes import IgnoredScope

MODEL_ID = "hf-internal-testing/tiny-random-OPTForCausalLM"
DEFAULT_RATIO = 0.4
DEFAULT_GROUP_SIZE = 4
DEFAULT_SENSITIVITY = nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION
DEFAULT_IGNORED_SCOPE = IgnoredScope()
DEFAULT_SUBSET_SIZE = 4
DEFAULT_MODE = nncf.CompressWeightsMode.INT4_ASYM


def create_transform_fn(model: OVModelForCausalLM, tokenizer: AutoTokenizer):
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
                shape[2 if shape[2].is_dynamic else 1] = 0
                inputs[input_name] = ov.Tensor(model_inputs.get_element_type(), shape.get_shape())
        return inputs

    return transform_fn


def _setup_model_and_dataset(model_id: str) -> Tuple[OVModelForCausalLM, nncf.Dataset]:
    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=False, compile=False, stateful=False)
    transform_fn = create_transform_fn(model, tokenizer)
    quantization_dataset = nncf.Dataset(dataset, partial(transform_fn))
    return model, quantization_dataset


def _test_basic_configurations(model, quantization_dataset, tmp_path, subset_size, mode) -> int:
    awq_options = [True, False]
    group_size_options = [1, 4]
    ratio_options = [0.4, 0.8]
    sensitivity_metrics = [
        nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
        nncf.SensitivityMetric.MAX_ACTIVATION_VARIANCE,
    ]
    ignored_scopes = [IgnoredScope(), IgnoredScope(types=["MatMul"])]

    load_count = 0
    for awq, group_size, ratio, sensitivity, scope in product(
        awq_options, group_size_options, ratio_options, sensitivity_metrics, ignored_scopes
    ):
        print(f"Testing: AWQ={awq}, Group={group_size}, Ratio={ratio}, Metric={sensitivity}, Scope={scope}")
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
            sensitivity_metric=sensitivity,
            ignored_scope=scope,
            lora_correction=False,
            advanced_parameters=AdvancedCompressionParameters(statistics_path=tmp_path / "statistics"),
        )
        load_count += 1
    return load_count


def _test_advanced_gptq_scale_estimation(model, quantization_dataset, tmp_path, subset_size, mode) -> int:
    load_count = 0
    for gptq, scale_est in product([True, False], [True, False]):
        print(f"Testing: AWQ=True, GPTQ={gptq}, Scale={scale_est}, LoRA=False")
        nncf.compress_weights(
            deepcopy(model.model),
            mode=mode,
            dataset=quantization_dataset,
            ratio=DEFAULT_RATIO,
            awq=True,
            gptq=gptq,
            group_size=DEFAULT_GROUP_SIZE,
            scale_estimation=scale_est,
            subset_size=subset_size,
            sensitivity_metric=DEFAULT_SENSITIVITY,
            ignored_scope=DEFAULT_IGNORED_SCOPE,
            lora_correction=False,
            advanced_parameters=AdvancedCompressionParameters(statistics_path=tmp_path / "statistics"),
        )
        load_count += 1
    return load_count


def _test_advanced_lora_scale_estimation(model, quantization_dataset, tmp_path, subset_size, mode) -> int:
    load_count = 0
    for scale_est, lora_corr in product([True, False], [True, False]):
        print(f"Testing: AWQ=True, GPTQ=False, Scale={scale_est}, LoRA={lora_corr}")
        nncf.compress_weights(
            deepcopy(model.model),
            mode=mode,
            dataset=quantization_dataset,
            ratio=DEFAULT_RATIO,
            awq=True,
            gptq=False,
            group_size=DEFAULT_GROUP_SIZE,
            scale_estimation=scale_est,
            subset_size=subset_size,
            sensitivity_metric=DEFAULT_SENSITIVITY,
            ignored_scope=DEFAULT_IGNORED_SCOPE,
            lora_correction=lora_corr,
            advanced_parameters=AdvancedCompressionParameters(statistics_path=tmp_path / "statistics"),
        )
        load_count += 1
    return load_count


def test_weight_compression_statistics_caching(tmp_path, mocker):
    """
    Tests the weight compression process, focusing on the statistics caching mechanism.
    Ensures that:
      - Statistics are collected once.
      - Statistics are loaded according to the number of configurations tested.
      - Statistics are dumped once.
    :param tmp_path: Temporary directory path for storing statistics.
    :param mocker: Mocking utility to spy on function calls.
    """
    from nncf.openvino.statistics.aggregator import OVStatisticsAggregator

    collect_spy = mocker.spy(OVStatisticsAggregator, "collect_statistics")
    load_spy = mocker.spy(OVStatisticsAggregator, "load_statistics_from_dir")
    dump_spy = mocker.spy(OVStatisticsAggregator, "dump_statistics")

    model_id = MODEL_ID
    subset_size = DEFAULT_SUBSET_SIZE
    mode = DEFAULT_MODE
    model, quantization_dataset = _setup_model_and_dataset(model_id)

    load_count = 0
    load_count += _test_basic_configurations(model, quantization_dataset, tmp_path, subset_size, mode)
    load_count += _test_advanced_gptq_scale_estimation(model, quantization_dataset, tmp_path, subset_size, mode)
    load_count += _test_advanced_lora_scale_estimation(model, quantization_dataset, tmp_path, subset_size, mode)

    assert collect_spy.call_count == 1, "Statistics should be collected only once."
    assert load_spy.call_count == load_count, f"Expected {load_count} load calls, found {load_spy.call_count}."
    assert dump_spy.call_count == 1, "Statistics should be dumped only once."
