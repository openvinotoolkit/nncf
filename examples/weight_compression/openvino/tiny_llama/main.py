import datetime
import itertools
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeVar

import numpy as np
import openvino as ov
from datasets import load_dataset
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from whowhatbench import Evaluator

import nncf
from nncf.common.logging import nncf_logger

DataItem = TypeVar("DataItem")
ModelInput = TypeVar("ModelInput")

ROOT = Path(__file__).parent.resolve()
MODEL_PATH = ROOT / "compressed_model.xml"

COMPRESSION_MODE = nncf.parameters.CompressWeightsMode.INT4_SYM
MAX_DROP = 0.2
MIN_GROUP_SIZE = 64
MAX_GROUP_SIZE = 128
MIN_RATIO = 0.5
MAX_RATIO = 1.0


def compress_model(ov_model: ov.Model, nncf_dataset: nncf.Dataset, ratio: float, group_size: int) -> ov.Model:
    """
    Compress the given OpenVINO model using NNCF weight compression.

    :param ov_model: The original OpenVINO model to be compressed.
    :param nncf_dataset: A representative dataset for the weight compression algorithm.
    :param ratio: The ratio between baseline and backup precisions
    :param group_size: Number of weights (e.g. 128) in the channel dimension
        that share quantization parameters (scale).
    :return: The OpenVINO model with compressed weights.
    """
    optimized_ov_model = nncf.compress_weights(
        ov_model.clone(), # we should clone the model because `compress_weights` works in place
        dataset=nncf_dataset,
        mode=COMPRESSION_MODE,
        ratio=ratio,
        group_size=group_size,
        sensitivity_metric=nncf.parameters.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
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
    group_size = optimized_model.get_rt_info()['nncf']['weight_compression']['group_size'].value
    ratio = optimized_model.get_rt_info()['nncf']['weight_compression']['ratio'].value
    nncf_logger.info(
        f"The similarity of model compressed with group_size={group_size}, ratio={ratio} is {similarity:.3f}"
    )
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


def print_results(optimized_model: ov.Model, ratio: float, group_size: int, similarity: float) -> None:
    """
    Print report with optimization details, memory footprint, and similarity score.

    :param optimized_model: The OpenVINO model with compressed weights.
    :param ratio: The ratio between baseline and backup precisions
    :param group_size: Number of weights (e.g. 128) in the channel dimension
        that share quantization parameters (scale).
    """
    ov.save_model(optimized_model, MODEL_PATH)
    footprint = Path(MODEL_PATH).with_suffix(".bin").stat().st_size
    print(f"Compressed model was saved to: {MODEL_PATH}")
    print(f"Best parameters: group_size={group_size}, ratio={ratio}")
    print(f"Memory footprint: {footprint / 2**20 :.2f} MB")
    print(f"Similarity: {similarity:.2f}")


def find_optimal_parameters(evaluator: Evaluator, model: OVModelForCausalLM, nncf_dataset: nncf.Dataset) -> None:
    """
    Find the optimal `ratio` and `group_size` for weight compression algorithm.

    :param evaluator: The evaluator object from whowhatbench Benchmark.
    :param model: The OpenVINO model for causal language modeling.
    :param nncf_dataset: A representative dataset for the weight compression algorithm.
    """
    original_ov_model = model.model
    evaluate_fn = partial(evaluate_model, hf_model=model, original_ov_model=original_ov_model, evaluator=evaluator)

    param_grid = list(itertools.product(list((x / 10 for x in range(10, 4, -1))), [MAX_GROUP_SIZE, MIN_GROUP_SIZE]))
    ratio, group_size = param_grid[0]  # (MAX_GROUP_SIZE, MAX_RATIO)
    optimized_model = compress_model(original_ov_model, nncf_dataset, ratio, group_size)
    similarity = evaluate_fn(optimized_model=optimized_model)
    if similarity >= 1 - MAX_DROP:
        nncf_logger.info(f"Compress embeddings and last layers to {COMPRESSION_MODE.value} precision")
        full_optimized_model = nncf.compress_weights(
            original_ov_model.clone(),
            mode=COMPRESSION_MODE,
            ratio=ratio,
            group_size=group_size,
            all_layers=True,
        )
        all_layers_similarity = evaluate_fn(optimized_model=full_optimized_model)
        if all_layers_similarity >= 1 - MAX_DROP:
            print_results(full_optimized_model, ratio, group_size, all_layers_similarity)
        else:
            print_results(optimized_model, ratio, group_size, similarity)
        return

    ratio, group_size = param_grid[-1]  # (MIN_GROUP_SIZE, MIN_RATIO)
    optimized_model = compress_model(original_ov_model, nncf_dataset, ratio, group_size)
    similarity = evaluate_fn(optimized_model=optimized_model)
    if similarity < 1 - MAX_DROP:
        nncf_logger.info(
            "The model was compressed with the minimum ratio and group_size, ",
            "but it could not achieve the required accuracy drop. ",
            "We recommend choosing a different mode for weight compression."
        )
        print_results(optimized_model, ratio, group_size, similarity)
        return

    for ratio, group_size in param_grid[1:-1]:
        optimized_model = compress_model(original_ov_model, nncf_dataset, ratio, group_size)
        similarity = evaluate_fn(optimized_model=optimized_model)
        if similarity >= 1 - MAX_DROP:
            print_results(optimized_model, ratio, group_size, similarity)
            return

    optimized_model = compress_model(original_ov_model, nncf_dataset, MIN_RATIO, MIN_GROUP_SIZE)
    print_results(optimized_model, MIN_RATIO, MIN_GROUP_SIZE, similarity)


def tiny_llama_transform_func(item, tokenizer):
    tokens = tokenizer(item["text"])
    attention_mask = np.expand_dims(np.array(tokens["attention_mask"]), 0)
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids = np.ma.array(position_ids, mask=attention_mask == 0)
    position_ids.filled(fill_value=1)
    res = {
        "input_ids": np.expand_dims(np.array(tokens["input_ids"]), 0),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    def gen_pkv(num_heads, head_dim, num_layers):
        res = {}
        for i in range(num_layers):
            res[f"past_key_values.{i}.key"] = np.zeros((1, num_heads, 0, head_dim))
            res[f"past_key_values.{i}.value"] = np.zeros((1, num_heads, 0, head_dim))
        return res

    res.update(gen_pkv(4, 64, 22))
    return res


model_id = "PY007/TinyLlama-1.1B-step-50K-105b"
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
model = OVModelForCausalLM.from_pretrained(
    model_id,
    export=True,
    trust_remote_code=True,
    use_cache=True,
    ov_config=ov_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = load_dataset("wikitext", "wikitext-2-v1", split="train[:1000]")
dataset = dataset.filter(lambda example: len(example["text"]) > 128)
transform_func = partial(tiny_llama_transform_func, tokenizer=tokenizer)

start = datetime.datetime.now()
evaluator = Evaluator(model, tokenizer=tokenizer, metrics=("similarity",))
nncf_dataset = get_nncf_dataset(dataset, transform_func)
find_optimal_parameters(evaluator, model, nncf_dataset)
end = datetime.datetime.now()
delta = end - start
delta -= datetime.timedelta(microseconds=delta.microseconds)
print(f"Elapsed time: {delta}")
