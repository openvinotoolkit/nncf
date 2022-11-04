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

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import nncf
import openvino.runtime as ov
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras.metrics import Metric

from examples.experimental.tensorflow.mobilenet_v2.utils import center_crop

# Path to the `mobilenet_v2` directory.
ROOT = Path(__file__).parent.resolve()
# Path to the directory where the original and quantized models will be saved.
MODEL_DIR = ROOT / 'mobilenet_v2_quantization'
# Path to ImageNet validation dataset.
DATASET_DIR = ROOT / 'imagenet'


def run_example():
    """
    Runs the MobileNetV2 quantization example.
    """
    # Step 1: Instantiate the MobileNetV2 from the Keras Applications.
    model = tf.keras.applications.MobileNetV2()

    # Step 2: Create calibration dataset.
    data_source = create_data_source(batch_size=128)

    # Step 3: Apply quantization algorithm.

    # Define the transformation method. This method should take a data item returned
    # per iteration through the `data_source` object and transform it into the model's
    # expected input that can be used for the model inference.
    def transform_fn(data_item):
        images, _ = data_item
        return images

    # Wrap framework-specific data source into the `nncf.Dataset` object.
    calibration_dataset = nncf.Dataset(data_source, transform_fn)

    # Quantization of the TensorFlow Keras model that was created using
    # Sequential or Keras Functional API. The `quantize` method expects
    # a TensorFlow Keras model as input and returns a TensorFlow Keras model
    # that has been quantized using the calibration dataset.
    quantized_model = nncf.quantize(model, calibration_dataset)

    # Step 4: Save the quantized TensorFlow Keras model.
    if not MODEL_DIR.exists():
        os.makedirs(MODEL_DIR)

    model_name = 'mobilenet_v2'
    saved_model_dir = MODEL_DIR / model_name
    quantized_model.save(saved_model_dir)
    print(f'The quantized model is exported to {saved_model_dir}')

    # Step 5: Run OpenVINO Model Optimizer to convert TensorFlow model to OpenVINO IR.
    mo_command = f'mo --saved_model_dir {saved_model_dir} ' \
                 f'--model_name {model_name} --output_dir {MODEL_DIR}'
    subprocess.call(mo_command, shell=True)

    # Step 6: Compare the accuracy of the original and quantized models.
    print('Checking the accuracy of the original model:')

    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name='acc@1'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='acc@5')
    ]

    model.compile(metrics=metrics)
    results = model.evaluate(data_source)
    print(f'The original model accuracy@top1: {results[1]:.4f}')

    print('Checking the accuracy of the quantized model:')
    ie = ov.Core()
    ir_model_xml = MODEL_DIR / f'{model_name}.xml'
    ir_model_bin = MODEL_DIR / f'{model_name}.bin'
    ir_quantized_model = ie.read_model(model=ir_model_xml, weights=ir_model_bin)
    quantized_compiled_model = ie.compile_model(ir_quantized_model, device_name='CPU')
    quantized_results = validate(quantized_compiled_model, metrics, data_source)
    print(f'The quantized model accuracy@top1: {quantized_results[0]:.4f}')


def create_data_source(batch_size: int) -> tf.data.Dataset:
    """
    Creates validation ImageNet dataset.

    :param batch_size: A number of elements return per iteration.
    :return: The instance of `tf.data.Dataset`.
    """
    val_dataset = tfds.load('imagenet2012', split='validation', data_dir=DATASET_DIR)

    def preprocess(data_item: Dict[str, tf.Tensor]):
        image = data_item['image']
        image = center_crop(image, 224, 32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = tf.image.convert_image_dtype(image, tf.float32)

        label = data_item['label']
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, 1000)
        label = tf.reshape(label, [1000])

        return image, label

    val_dataset = val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size)

    return val_dataset


def validate(model: ov.Model,
             metrics: List[Metric],
             val_dataset: tf.data.Dataset,
             print_freq: int = 100) -> Tuple[float]:
    """
    Validate the accuracy of the model.

    :param model: An OpenVINO model.
    :param metrics: A list of TensorFlow metrics
    :param val_dataset: An instant of the TensorFlow dataset
    :param print_freq: A print frequency (batch iterations).
    :return: A Tuple of scalars of computed metrics
    """
    for m in metrics:
        m.reset_state()

    num_items = len(val_dataset)
    for i, (images, labels) in enumerate(val_dataset):
        input_data = images.numpy()

        logit = model(input_data)
        pred = list(logit.values())[0]

        for m in metrics:
            m.update_state(labels, pred)
        if i % print_freq == 0 or i + 1 == num_items:
            output = [f'{i + 1}/{num_items}:']
            for m in metrics:
                  output.append(f'{m.name}: {m.result().numpy():.4f}')
            print(' '.join(output))

    return metrics[0].result().numpy(), metrics[1].result().numpy()


if __name__ == '__main__':
    run_example()
