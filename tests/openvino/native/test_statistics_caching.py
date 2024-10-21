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
import numpy as np
import openvino as ov
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.common.graph.transformations.commands import TargetType
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.statistics.aggregator import OVStatisticsAggregator
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from tests.cross_fw.test_templates.test_statistics_caching import TemplateTestStatisticsCaching
from tests.openvino.native.models import AWQMatmulModel


class TestStatisticsCaching(TemplateTestStatisticsCaching):
    def create_target_point(self, target_point_type: TargetType, name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_point_type, name, port_id)

    def get_statistics_aggregator(self):
        return OVStatisticsAggregator(None)


def test_cached_configuration_tinyllama(mocker):
    from nncf.openvino.statistics.aggregator import OVStatisticsAggregator

    collect_statistics_spy = mocker.spy(OVStatisticsAggregator, "collect_statistics")
    # load_statistics_from_file_spy = mocker.spy(OVStatisticsAggregator, "load_statistics_from_file")
    dump_statistics_spy = mocker.spy(OVStatisticsAggregator, "dump_statistics")

    MODEL_ID = "PY007/TinyLlama-1.1B-Chat-v0.3"

    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True, load_in_8bit=False, compile=False, stateful=False)

    def transform_fn(data, model, tokenizer):
        tokenized_text = tokenizer(data["text"], return_tensors="np")
        input_ids = tokenized_text["input_ids"]
        attention_mask = tokenized_text["attention_mask"]

        inputs = {}
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = tokenized_text["attention_mask"]
        position_ids = np.cumsum(attention_mask, axis=1) - 1
        position_ids[attention_mask == 0] = 1

        # The magic forms KV cache as model inputs
        batch_size = input_ids.shape[0]
        for input_name in model.key_value_input_names:
            model_inputs = model.model.input(input_name)
            shape = model_inputs.get_partial_shape()
            shape[0] = batch_size
            if shape[2].is_dynamic:
                shape[2] = 0
            else:
                shape[1] = 0
            inputs[input_name] = ov.Tensor(model_inputs.get_element_type(), shape.get_shape())

        inputs["position_ids"] = position_ids
        return inputs

    quantization_dataset = nncf.Dataset(dataset, partial(transform_fn, model=model, tokenizer=tokenizer))

    # gptq = [True, False]
    awq = [True, False]
    scale_estimation = [True, False]
    group_size = [1, 64]
    ratio = [0.4, 0.8]
    sensitivity_metric = [
        nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
        nncf.SensitivityMetric.MEAN_ACTIVATION_VARIANCE,
        nncf.SensitivityMetric.MAX_ACTIVATION_VARIANCE,
        nncf.SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE,
    ]
    all_combinations = product(awq, scale_estimation, group_size, ratio, sensitivity_metric)
    print(all_combinations)
    for params in all_combinations:
        param_dict = {
            # "gptq": params[0],
            "awq": params[0],
            "scale_estimation": params[1],
            "group_size": params[2],
            "ratio": params[3],
            "sensitivity_metric": params[4],
        }

        _ = nncf.compress_weights(
            model.model,
            mode=nncf.CompressWeightsMode.INT4_ASYM,
            dataset=quantization_dataset,
            ratio=param_dict["ratio"],
            # gptq=param_dict["gptq"],
            awq=param_dict["awq"],
            group_size=param_dict["group_size"],
            scale_estimation=param_dict["scale_estimation"],
            subset_size=128,
            sensitivity_metric=param_dict["sensitivity_metric"],
            advanced_parameters=AdvancedCompressionParameters(statistics_file_path="./full_stat"),
        )

    assert collect_statistics_spy.call_count == 1, "Statistics should only be collected once"
    # assert load_statistics_from_file_spy.call_count == 2, "Statistics should be loaded from file twice."
    assert dump_statistics_spy.call_count == 1, "Statistics should only be dumped once."


def test_weight_compression_caching_e2e(tmp_path, mocker):
    """
    Tests the caching behavior of weight compression with different sensitivity metrics.
    Ensures that statistics are collected, dumped, and loaded correctly during compression.

    :param tmp_path: Temporary directory path for storing statistics.
    :param mocker: Mocker fixture for spying on methods.
    :param sensitivity_metric: Sensitivity metric for weight compression.
    """
    from nncf.openvino.statistics.aggregator import OVStatisticsAggregator

    collect_statistics_spy = mocker.spy(OVStatisticsAggregator, "collect_statistics")
    load_statistics_from_file_spy = mocker.spy(OVStatisticsAggregator, "load_statistics_from_file")
    dump_statistics_spy = mocker.spy(OVStatisticsAggregator, "dump_statistics")

    # dataset_size = 4
    model = AWQMatmulModel().ov_model
    sz = 8
    n_samples = 10
    dataset = nncf.Dataset([np.ones([1, i + 1, sz]) for i in range(n_samples)])

    test_file = "statistics"

    awq = [True, False]
    scale_estimation = [True, False]
    group_size = [1, 2, 4]
    ratio = [0.4, 0.8, 0.9]
    sensitivity_metric = [
        nncf.SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,
        nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
        nncf.SensitivityMetric.MEAN_ACTIVATION_VARIANCE,
        nncf.SensitivityMetric.MAX_ACTIVATION_VARIANCE,
        nncf.SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE,
    ]
    gptq = [True, False]
    all_combinations = product(awq, scale_estimation, group_size, ratio, sensitivity_metric, gptq)
    cnt = 0
    for params in all_combinations:
        print(params)
        param_dict = {
            "awq": params[0],
            "scale_estimation": params[1],
            "group_size": params[2],
            "ratio": params[3],
            "sensitivity_metric": params[4],
            "gptq": params[5],
        }

        _ = nncf.compress_weights(
            deepcopy(model),
            mode=nncf.CompressWeightsMode.INT4_ASYM,
            dataset=dataset,
            ratio=param_dict["ratio"],
            gptq=param_dict["gptq"],
            awq=param_dict["awq"],
            group_size=param_dict["group_size"],
            scale_estimation=param_dict["scale_estimation"],
            subset_size=128,
            sensitivity_metric=param_dict["sensitivity_metric"],
            advanced_parameters=AdvancedCompressionParameters(statistics_file_path=tmp_path / test_file),
        )
        cnt += 1

    assert collect_statistics_spy.call_count == 1, "Statistics should only be collected once"
    assert load_statistics_from_file_spy.call_count == cnt, "Statistics should be loaded from file twice."
    assert dump_statistics_spy.call_count == 1, "Statistics should only be dumped once."
