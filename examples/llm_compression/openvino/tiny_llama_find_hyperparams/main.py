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

import datetime
import itertools
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple, TypeVar

import numpy as np
import openvino as ov
from datasets import load_dataset
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from whowhatbench import Evaluator

import nncf
from nncf.common.logging import nncf_logger
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters

DataItem = TypeVar("DataItem")
ModelInput = TypeVar("ModelInput")

ROOT = Path(__file__).parent.resolve()
MODEL_PATH = ROOT / "compressed_model.xml"
STATISTICS_PATH = ROOT / "statistics"

COMPRESSION_MODE = nncf.parameters.CompressWeightsMode.INT4_SYM
MAX_DROP = 0.2
# We consider the following range of parameters: group_size - [64, 128], ratio - [0.5,...,1.0]
MIN_GROUP_SIZE = 64
MAX_GROUP_SIZE = 128
MIN_RATIO = 0.5
MAX_RATIO = 1.0
RATIO_STEP = 0.1


def compress_model(
    ov_model: ov.Model, nncf_dataset: nncf.Dataset, ratio: float, group_size: int, awq: bool
) -> ov.Model:
    """
    Compress the given OpenVINO model using NNCF weight compression.

    :param ov_model: The original OpenVINO model to be compressed.
    :param nncf_dataset: A representative dataset for the weight compression algorithm.
    :param ratio: The ratio between baseline and backup precisions
    :param group_size: Number of weights (e.g. 128) in the channel dimension
        that share quantization parameters (scale).
    :param awq: Indicates whether use AWQ weights correction.
    :return: The OpenVINO model with compressed weights.
    """
    optimized_ov_model = nncf.compress_weights(
        ov_model.clone(),  # we should clone the model because `compress_weights` works in place
        dataset=nncf_dataset,
        mode=COMPRESSION_MODE,
        ratio=ratio,
        group_size=group_size,
        awq=awq,
        sensitivity_metric=nncf.parameters.SensitivityMetric.MAX_ACTIVATION_VARIANCE,
        advanced_parameters=AdvancedCompressionParameters(statistics_path=STATISTICS_PATH),
    )
    return optimized_ov_model


def evaluate_model(
    hf_model: OVModelForCausalLM, original_ov_model: ov.Model, optimized_model: ov.Model, evaluator: Evaluator
) -> float:
    """
    Get similarity of text generation between original and optimized models.

    :param hf_model: The OpenVINO model for causal language modeling.
    :param original_ov_model: The original OpenVINO model
    :param optimized_model: The OpenVINO model with compressed weights.
    :param evaluator: The evaluator object from whowhatbench Benchmark.
    :return: The similarity score between the original and optimized models.
    """
    hf_model.model = optimized_model
    hf_model.request = None
    _, all_metrics = evaluator.score(hf_model)
    hf_model.model = original_ov_model.clone()
    hf_model.request = None
    similarity = all_metrics["similarity"][0]
    group_size = optimized_model.get_rt_info()["nncf"]["weight_compression"]["group_size"].value
    ratio = float(optimized_model.get_rt_info()["nncf"]["weight_compression"]["ratio"].value)
    awq = optimized_model.get_rt_info()["nncf"]["weight_compression"]["awq"].value
    all_layers = optimized_model.get_rt_info()["nncf"]["weight_compression"]["all_layers"].value
    params_info = f"The similarity of model compressed with group_size={group_size}, ratio={ratio:.1f}, awq={awq}"
    if all_layers == "True":
        params_info = params_info + ", all_layers=True"
    nncf_logger.info(params_info + f" is {similarity:.3f}")
    return similarity


def get_nncf_dataset(
    data_source: Iterable[DataItem], transform_func: Optional[Callable[[DataItem], ModelInput]] = None
) -> nncf.Dataset:
    """
    Create an NNCF dataset for the weight compression algorithm.

    :param data_source: The iterable object serving as the source of data items.
    :param transform_func: The transformation function applied to the data_source.
    :return: nncf_dataset: NNCF Dataset for the weight compression algorithm.
    """
    if data_source is None:
        return None
    if transform_func:
        return nncf.Dataset(data_source, transform_func)
    return nncf.Dataset(data_source)


def print_results(optimized_model: ov.Model, similarity: float) -> None:
    """
    Print report with optimization details, memory footprint, and similarity score.

    :param optimized_model: The OpenVINO model with compressed weights.
    :param similarity: The similarity score between the original and optimized models.
    """
    ov.save_model(optimized_model, MODEL_PATH)
    print(f"Compressed model was saved to: {MODEL_PATH}")
    group_size = optimized_model.get_rt_info()["nncf"]["weight_compression"]["group_size"].value
    ratio = float(optimized_model.get_rt_info()["nncf"]["weight_compression"]["ratio"].value)
    awq = optimized_model.get_rt_info()["nncf"]["weight_compression"]["awq"].value
    all_layers = optimized_model.get_rt_info()["nncf"]["weight_compression"]["all_layers"].value
    best_params_info = f"Best parameters: group_size={group_size}, ratio={ratio:.1f}, awq={awq}"
    if all_layers == "True":
        print(best_params_info + ", all_layers=True")
    else:
        print(best_params_info)
    footprint = Path(MODEL_PATH).with_suffix(".bin").stat().st_size
    print(f"Memory footprint: {footprint / 2**20 :.2f} MB")
    print(f"Similarity: {similarity:.2f}")


def find_parameters(
    evaluator: Evaluator, model: OVModelForCausalLM, nncf_dataset: nncf.Dataset
) -> Tuple[bool, float, int]:
    """
    Find the optimal `awq`, `ratio` and `group_size` for weight compression algorithm.

    :param evaluator: The evaluator object from whowhatbench Benchmark.
    :param model: The OpenVINO model for causal language modeling.
    :param nncf_dataset: A representative dataset for the weight compression algorithm.
    :return: The optimal awq, ratio and group_size.
    """
    original_ov_model = model.model
    evaluate_fn = partial(evaluate_model, hf_model=model, original_ov_model=original_ov_model, evaluator=evaluator)

    # Generating a grid of hyperparameter values for tuning, combining ratios and group sizes.
    ratio_grid = np.arange(MAX_RATIO, MIN_RATIO - RATIO_STEP, -RATIO_STEP)
    param_grid = list(itertools.product(ratio_grid, [MAX_GROUP_SIZE, MIN_GROUP_SIZE]))

    # First, we try to use the maximum ratio and group_size to get the most efficient model
    ratio, group_size = param_grid[0]  # (MAX_GROUP_SIZE, MAX_RATIO)
    use_awq = False
    optimized_model = compress_model(original_ov_model, nncf_dataset, ratio, group_size, awq=use_awq)
    similarity = evaluate_fn(optimized_model=optimized_model)
    if similarity >= 1 - MAX_DROP:
        # If model with the maximum ratio and group_size is acceptable,
        # we try to compress embeddings and last MatMul layers to a primary precision
        full_optimized_model = nncf.compress_weights(
            original_ov_model.clone(),
            mode=COMPRESSION_MODE,
            ratio=ratio,
            group_size=group_size,
            all_layers=True,
        )
        all_layers_similarity = evaluate_fn(optimized_model=full_optimized_model)
        if all_layers_similarity >= 1 - MAX_DROP:
            print_results(full_optimized_model, all_layers_similarity)
        else:
            print_results(optimized_model, similarity)
        return use_awq, ratio, group_size

    # If the best performing model is not acceptable, we try to use AWQ weights correction and compare similarity
    use_awq = True
    optimized_model = compress_model(original_ov_model, nncf_dataset, ratio, group_size, awq=use_awq)
    awq_similarity = evaluate_fn(optimized_model=optimized_model)
    if awq_similarity >= 1 - MAX_DROP:
        print_results(optimized_model, awq_similarity)
        return use_awq, ratio, group_size
    use_awq = awq_similarity > similarity

    # If the best performing model is not acceptable, we try to use the smallest ratio and group_size
    # to check the reachability of the max drop criterion
    ratio, group_size = param_grid[-1]  # (MIN_GROUP_SIZE, MIN_RATIO)
    optimized_model = compress_model(original_ov_model, nncf_dataset, ratio, group_size, awq=use_awq)
    similarity = evaluate_fn(optimized_model=optimized_model)
    if similarity < 1 - MAX_DROP:
        nncf_logger.info(
            "The model was compressed with the minimum ratio and group_size, "
            "but it could not achieve the required accuracy drop. "
            "We recommend choosing a different mode for weight compression."
        )
        print_results(optimized_model, similarity)
        return use_awq, ratio, group_size

    # If max drop criterion is achivable, we run a grid-search to find the best parameters
    for ratio, group_size in param_grid[1:-1]:
        optimized_model = compress_model(original_ov_model, nncf_dataset, ratio, group_size, awq=use_awq)
        similarity = evaluate_fn(optimized_model=optimized_model)
        if similarity >= 1 - MAX_DROP:
            print_results(optimized_model, similarity)
            return use_awq, ratio, group_size

    optimized_model = compress_model(original_ov_model, nncf_dataset, MIN_RATIO, MIN_GROUP_SIZE, awq=use_awq)
    print_results(optimized_model, similarity)
    return use_awq, MIN_RATIO, MIN_GROUP_SIZE


def tiny_llama_transform_func(item, tokenizer, ov_model):  # <YOUR_TRANSFORMATION_FUNCTION>
    input_dtypes = {inp.get_any_name(): inp.get_element_type() for inp in ov_model.inputs}
    tokens = tokenizer(item["text"])
    input_ids = np.expand_dims(np.array(tokens["input_ids"]), 0)
    attention_mask = np.expand_dims(np.array(tokens["attention_mask"]), 0)
    position_ids = np.cumsum(attention_mask, axis=1) - 1
    position_ids[attention_mask == 0] = 1
    res = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids.reshape(*attention_mask.shape),
    }

    def gen_pkv(num_heads, head_dim, num_layers):
        res = {}
        shape = (1, num_heads, 0, head_dim)
        for i in range(num_layers):
            key_name = f"past_key_values.{i}.key"
            val_name = f"past_key_values.{i}.value"
            res[key_name] = ov.Tensor(shape=shape, type=input_dtypes[key_name])
            res[val_name] = ov.Tensor(shape=shape, type=input_dtypes[val_name])
        return res

    res.update(gen_pkv(4, 64, 22))
    return res


def main():
    model_id = "TinyLlama/TinyLlama-1.1B-step-50K-105b"  # <YOUR_MODEL_ID>
    ov_config = {
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": "1",
        "CACHE_DIR": "",
        "DYNAMIC_QUANTIZATION_GROUP_SIZE": "0",
        "KV_CACHE_PRECISION": "f16",
    }
    model = OVModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        trust_remote_code=True,
        use_cache=True,
        ov_config=ov_config,
        stateful=False,
        load_in_8bit=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train[:1000]")  # <YOUR_DATASET>
    dataset = dataset.filter(lambda example: len(example["text"]) > 128)
    transform_func = partial(tiny_llama_transform_func, tokenizer=tokenizer, ov_model=model.model)

    start = datetime.datetime.now()
    evaluator = Evaluator(model, tokenizer=tokenizer, metrics=("similarity",))
    nncf_dataset = get_nncf_dataset(dataset, transform_func)
    awq, ratio, group_size = find_parameters(evaluator, model, nncf_dataset)
    end = datetime.datetime.now()
    delta = end - start
    delta -= datetime.timedelta(microseconds=delta.microseconds)
    print(f"Elapsed time: {delta}")
    return awq, ratio, group_size


if __name__ == "__main__":
    main()
