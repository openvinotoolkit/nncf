"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import Iterable
from typing import Any

import subprocess
from pathlib import Path

import openvino.runtime as ov
import torch
import nncf
import datasets
import evaluate
import transformers
import numpy as np


# Path to the `bert` directory.
ROOT = Path(__file__).parent.resolve()
# Path to the directory where the original and quantized IR will be saved.
MODEL_DIR = ROOT / 'bert_quantization'
# Path to the pre-trained model directory.
PRETRAINED_MODEL_DIR = ROOT / 'MRPC'

TASK_NAME = 'mrpc'
MAX_SEQ_LENGTH = 128


ie = ov.Core()


def run_example():
    """
    Runs the BERT quantization example.
    """
    # Step 1: Load pre-trained model.
    model = transformers.BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_DIR)
    model.eval()

    # Step 2: Converts PyTorch model to the OpenVINO model.
    ov_model = convert_torch_to_openvino(model)

    # Step 3: Create calibration dataset.
    data_source = create_data_source()

    # Step 4: Apply quantization algorithm.

    # Define the transformation method. This method should take a data item returned
    # per iteration through the `data_source` object and transform it into the model's
    # expected input that can be used for the model inference.
    INPUT_NAMES = [x.any_name for x in ov_model.inputs]
    def transform_fn(data_item):
        inputs = {
            name: np.asarray(data_item[name], dtype=np.int64) for name in INPUT_NAMES
        }
        return [inputs]

    # Wrap framework-specific data source into the `nncf.Dataset` object.
    calibration_dataset = nncf.Dataset(data_source, transform_fn)
    quantized_model = nncf.quantize(ov_model, calibration_dataset, model_type='transformer')

    # Step 5: Save quantized model.
    ir_qmodel_xml = MODEL_DIR / 'bert_base_mrpc_quantized.xml'
    ir_qmodel_bin = MODEL_DIR / 'bert_base_mrpc_quantized.bin'
    ov.serialize(quantized_model, str(ir_qmodel_xml), str(ir_qmodel_bin))

    # Step 6: Compare the accuracy of the original and quantized models.
    print('Checking the accuracy of the original model:')
    metric = validation_fn(ov_model, data_source)
    print(f'F1 score: {metric}')

    print('Checking the accuracy of the quantized model:')
    metric = validation_fn(quantized_model, data_source)
    print(f'F1 score: {metric}')

    # Step 7: Compare Performance of the original and quantized models.
    # benchmark_app -m bert_quantization/bert_base_mrpc.xml -d CPU -api async
    # benchmark_app -m bert_quantization/bert_base_mrpc_quantized.xml -d CPU -api async


def convert_torch_to_openvino(model: torch.nn.Module) -> ov.Model:
    """
    Converts the fine-tuned BERT model for the MRPC task to the OpenVINO IR format.
    """
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir()

    # Export PyTorch model to the ONNX format.
    onnx_model_path = MODEL_DIR / 'bert_base_mrpc.onnx'
    dummy_input = (
        torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.int64),  # input_ids
        torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.int64),  # attention_mask
        torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.int64),  # token_type_ids
    )
    torch.onnx.export(model,
                      dummy_input,
                      onnx_model_path,
                      verbose=False,
                      opset_version=11,
                      input_names=['input_ids', 'attention_mask', 'token_type_ids'],
                      output_names=['output'])

    # Run Model Optimizer to convert ONNX model to OpenVINO IR.
    mo_command = f'mo --framework onnx -m {onnx_model_path} --output_dir {MODEL_DIR}'
    subprocess.call(mo_command, shell=True)

    ir_model_xml = MODEL_DIR / 'bert_base_mrpc.xml'
    ir_model_bin = MODEL_DIR / 'bert_base_mrpc.bin'
    ov_model = ie.read_model(model=ir_model_xml, weights=ir_model_bin)

    return ov_model


def create_data_source() -> datasets.Dataset:
    """
    Creates validation MRPC dataset.

    :return: The `datasets.Dataset` object.
    """
    raw_dataset = datasets.load_dataset('glue', TASK_NAME, split='validation')
    tokenizer = transformers.BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)

    def _preprocess_fn(examples):
        texts = (examples['sentence1'], examples['sentence2'])
        result = tokenizer(*texts, padding='max_length', max_length=MAX_SEQ_LENGTH, truncation=True)
        result['labels'] = examples['label']
        return result
    processed_dataset = raw_dataset.map(_preprocess_fn, batched=True, batch_size=1)

    return processed_dataset


def validation_fn(model: ov.Model, validation_dataset: Iterable[Any]) -> float:
    compiled_model = ie.compile_model(model, device_name='CPU')
    output_layer = compiled_model.output(0)

    metric = evaluate.load('glue', TASK_NAME)
    INPUT_NAMES = [x.any_name for x in compiled_model.inputs]
    for batch in validation_dataset:
        inputs = [
            np.expand_dims(np.asarray(batch[key], dtype=np.int64), 0) for key in INPUT_NAMES
        ]
        outputs = compiled_model(inputs)[output_layer]
        predictions = outputs[0].argmax(axis=-1)
        metric.add_batch(predictions=[predictions], references=[batch['labels']])
    metrics = metric.compute()
    f1_score = metrics['f1']

    return f1_score


if __name__ == '__main__':
    run_example()
