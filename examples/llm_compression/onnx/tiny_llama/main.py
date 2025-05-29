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

import time
from pathlib import Path

import onnx
from optimum.intel.openvino import OVModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.onnx.quantization.backend_parameters import BackendParameters

ROOT = Path(__file__).parent.resolve()


MODEL_ID = "PY007/TinyLlama-1.1B-Chat-v0.3"
OUTPUT_DIR = ROOT / "tinyllama_compressed"


def main():
    # Export the pretrained model in ONNX format. The OUTPUT_DIR directory
    # will contain model.onnx, model.onnx_data, and some metadata files.
    model = ORTModelForCausalLM.from_pretrained(MODEL_ID, export=True)
    model.save_pretrained(OUTPUT_DIR)

    # Load the exported pretrained model as an ONNX model. For models larger than 2GB,
    # set `load_external_data=False` to load only the model's topology without the weights.
    # The weights will be loaded on the fly during compression. To enable this, specify the
    # `BackendParameters.EXTERNAL_DATA_DIR` parameter, which should be the absolute path to
    # the directory containing the model’s external data files.
    onnx_model = onnx.load(OUTPUT_DIR / "model.onnx", load_external_data=False)

    compressed_onnx_model = nncf.compress_weights(
        onnx_model,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=0.8,
        advanced_parameters=nncf.AdvancedCompressionParameters(
            backend_params={BackendParameters.EXTERNAL_DATA_DIR: OUTPUT_DIR}
        ),
    )

    # Replace the original model with the compressed model.
    onnx.save(compressed_onnx_model, OUTPUT_DIR / "model.onnx", save_as_external_data=True)

    # Infer Model.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    ov_model = OVModelForCausalLM.from_pretrained(
        OUTPUT_DIR,
        trust_remote_code=True,
        load_in_8bit=False,
        compile=False,
        stateful=False,
        ov_config={"DYNAMIC_QUANTIZATION_GROUP_SIZE": "0", "KV_CACHE_PRECISION": "f16"},
        export=False,
        from_onnx=True,
    )

    input_ids = tokenizer("What is PyTorch?", return_tensors="pt").to(device=model.device)

    start_t = time.time()
    output = ov_model.generate(**input_ids, max_new_tokens=100)
    print("Elapsed time: ", time.time() - start_t)

    output_text = tokenizer.decode(output[0])
    print(output_text)
    return output_text


if __name__ == "__main__":
    main()
