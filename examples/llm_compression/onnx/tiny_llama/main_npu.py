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
from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer

import nncf
from nncf.onnx.quantization.backend_parameters import BackendParameters

ROOT = Path(__file__).parent.resolve()

MODEL_ID = "microsoft/Phi-3.5-mini-instruct" # r"E:\download\huggingface\Phi-3.5-mini-instruct"# # 
OUTPUT_DIR = ROOT / "Phi-3.5-mini-instruct-onnx-21-symint4" # r"E:\download\onnx\full_support_op_models\Phi-3.5-mini-instruct-onnx-export-21-symint4"

def main():
    # This is a workaround for the NPU to handle quantized models correctly.
    # In current nncf:
    #   model with opset < 21  will use MatMulNBits OP, which is not supported by NPU.
    #   model with opset >= 21 will use DequantizeLinear OP, which is supported by NPU.
    # The main_export function will export the model with opset 20, and then modified to 21.
    # This is a temporary solution until the NPU supports the MatMulNBits OP.
    print("Exporting ONNX model ...")
    main_export(MODEL_ID, OUTPUT_DIR, task="text-generation-with-past", opset=20, no_post_process=True)

    # Load the exported pretrained model as an ONNX model. For models larger than 2GB,
    # set `load_external_data=False` to load only the model's topology without the weights.
    # The weights will be loaded on the fly during compression. To enable this, specify the
    # `BackendParameters.EXTERNAL_DATA_DIR` parameter, which should be the absolute path to
    # the directory containing the model’s external data files.
    print("Compressing ONNX model ...")
    onnx_model = onnx.load(OUTPUT_DIR / "model.onnx", load_external_data=False)
    onnx_model.ir_version = 10
    onnx_model.opset_import[0].version = 21

    compressed_onnx_model = nncf.compress_weights(
        onnx_model,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=1.0,
        all_layers=True,
        ignored_scope=nncf.IgnoredScope(types=["Gather"]),
        advanced_parameters=nncf.AdvancedCompressionParameters(
            backend_params={BackendParameters.EXTERNAL_DATA_DIR: OUTPUT_DIR}
        ),
    )

    # Replace the original model with the compressed model.
    print("Saving Compressed ONNX model ...")
    onnx.save(compressed_onnx_model, OUTPUT_DIR / "model.onnx", save_as_external_data=True)

    # Infer Model.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Testing Compressed ONNX model ...")
    ov_model = OVModelForCausalLM.from_pretrained(OUTPUT_DIR, from_onnx=True)

    input_ids = tokenizer("What is PyTorch?", return_tensors="pt").to(device="cpu")

    start_t = time.time()
    output = ov_model.generate(**input_ids, max_new_tokens=100)
    print("Elapsed time: ", time.time() - start_t)

    output_text = tokenizer.decode(output[0])
    print(output_text)
    return output_text


if __name__ == "__main__":
    main()
