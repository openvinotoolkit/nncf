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
from functools import partial

import datasets
import numpy as np
import openvino as ov
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer
from whowhatbench import Evaluator

import nncf


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


def main():
    MODEL_ID = "HuggingFaceTB/SmolLM-360M"
    OUTPUT_DIR = "smollm_360m_compressed"

    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Filtering to remove empty samples from the dataset
    dataset = dataset.filter(lambda example: len(example["text"]) > 1)

    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model = OVModelForCausalLM.from_pretrained(
        MODEL_ID,
        export=True,
        load_in_8bit=False,
        compile=False,
        stateful=False,
        ov_config={"INFERENCE_PRECISION_HINT": "f32"},
    )
    original_ov_model = hf_model.model.clone()

    evaluator = Evaluator(hf_model, hf_tokenizer, metrics=("similarity",))

    quantization_dataset = nncf.Dataset(dataset, partial(transform_fn, model=hf_model, tokenizer=hf_tokenizer))

    hf_model.model = nncf.quantize(
        hf_model.model,
        calibration_dataset=quantization_dataset,
        # Only PERFORMANCE preset supports in combination with FP8 quantization mode
        preset=nncf.QuantizationPreset.PERFORMANCE,
        mode=nncf.QuantizationMode.FP8_E4M3,
        model_type=nncf.ModelType.TRANSFORMER,
        # SmoothQuant algorithm is not needed for FP8 quantization
        advanced_parameters=nncf.AdvancedQuantizationParameters(
            smooth_quant_alphas=nncf.AdvancedSmoothQuantParameters(matmul=-1)
        ),
    )
    hf_model.save_pretrained(OUTPUT_DIR)
    hf_tokenizer.save_pretrained(OUTPUT_DIR)

    hf_model.model = original_ov_model

    optimized_hf_model = OVModelForCausalLM.from_pretrained(
        OUTPUT_DIR, ov_config={"DYNAMIC_QUANTIZATION_GROUP_SIZE": "0", "INFERENCE_PRECISION_HINT": "f32"}
    )

    _, all_metrics = evaluator.score(optimized_hf_model)
    similarity = all_metrics["similarity"][0]
    print(f"Similarity: {similarity}")
    return similarity


if __name__ == "__main__":
    main()
