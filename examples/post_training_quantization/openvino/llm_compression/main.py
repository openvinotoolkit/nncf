import time
from functools import partial

import openvino as ov
import numpy as np
from transformers import AutoTokenizer

import datasets
from optimum.intel.openvino import OVModelForCausalLM
import nncf


MODEL_ID = "PY007/TinyLlama-1.1B-Chat-v0.3"
OUTPUT_DIR = "tinyllama_compressed"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True, use_cache=True, compile=False)

dataset = datasets.load_dataset("allenai/c4", "allenai--c4", 
                data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                split="validation")

def transform_fn(data, model):
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

quantization_dataset = nncf.Dataset(dataset, partial(transform_fn, model=model))

model.model = nncf.compress_weights(model.model, dataset=quantization_dataset, mode=nncf.CompressWeightsMode.INT4_SYM, sensitivity_metric=nncf.parameters.SensitivityMetric.HESSIAN_INPUT_ACTIVATION)
model.save_pretrained(OUTPUT_DIR)

model = OVModelForCausalLM.from_pretrained(OUTPUT_DIR)
input_ids = tokenizer("What is PyTorch?", return_tensors="pt").to(device=model.device)

start_t = time.time()
output = model.generate(**input_ids, max_new_tokens=100)
print("Elapsed time: ", time.time() - start_t)

print(tokenizer.decode(output[0]))