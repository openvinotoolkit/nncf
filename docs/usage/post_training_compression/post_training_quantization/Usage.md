# Post-Training Quantization

Post-Training Quantization is a quantization algorithm that doesn't demand retraining of a quantized model.
It utilizes a small subset of the initial dataset to calibrate quantization constants.
Please refer to this [document](/docs/usage/training_time_compression/other_algorithms/LegacyQuantization.md) for details of the implementation.

NNCF provides an advanced Post-Training Quantization algorithm, which consists of the following techniques:

1) MinMaxQuantization - Analyzes the model and inserts extra quantization layers calibrated on a small subset.
2) FastBiasCorrection or BiasCorrection - Reduces the bias errors between quantized layers and the corresponding
   original layers.

## Usage

To start the algorithm, provide the following entities:

* Original model.
* Validation part of the dataset.
* [Data transformation function](#data-transformation-function) transforming data items from the original dataset to the model input data.

The basic workflow steps:

1. Create the [data transformation function](#data-transformation-function).

    ```python
    def transform_fn(data_item):
        images, _ = data_item
        return images
    ```

2. Create an instance of `nncf.Dataset` class by passing two parameters:

    * `data_source` - Iterable python object that contains data items for model calibration.
    * `transform_fn` - Data transformation function from the Step 1.

    ```python
    calibration_dataset = nncf.Dataset(val_dataset, transform_fn)
    ```

3. Run the quantization pipeline.

    ```python
    quantized_model = nncf.quantize(model, calibration_dataset)
    ```

## Data Transformation Function

Model input structure differs from one pipeline to another. Thus NNCF introduces the interface to adapt the user dataset format to the NNCF format. This interface is called the data transformation function.

Every backend has its own return value format for the data transformation function. It is based on the input structure of the
backend inference framework.
Below are the formats of data transformation function for each supported backend.

<details><summary><b>PyTorch, TorchFX, TensorFlow, OpenVINO</b></summary>

The return format of the data transformation function is directly the input tensors consumed by the model. \
_If you are not sure that your implementation of data transformation function is correct you can validate it by using the
following code:_

```python
model = ...  # Model
val_loader = ...  # Original Dataset
transform_fn = ...  # Data transformation function
for data_item in val_loader:
    model(transform_fn(data_item))
```

</details>
<details><summary><b>ONNX</b></summary>

[ONNX Runtime](https://onnxruntime.ai/) is used as the inference engine for the ONNX backend. \
The Input format of the data is the following - ```Dict[str, np.ndarray]```, where keys of the dictionary are the model input names and values are numpy tensors passed to these inputs.

_If you are not sure that your implementation of data transformation function is correct, you can validate it by using the
following code:_

```python
import onnxruntime

model_path = ...  # Path to Model
val_loader = ...  # Original Dataset
transform_fn = ...  # Data transformation function
sess = onnxruntime.InferenceSession(model_path)
output_names = [output.name for output in sess.get_outputs()]
for data_item in val_loader:
    sess.run(output_names, input_feed=transform_fn(data_item))
```

</details>

NNCF provides the examples of Post-Training Quantization where you can find the implementation of data transformation
function: [PyTorch](/examples/post_training_quantization/torch/mobilenet_v2/README.md), [TorchFX](/examples/post_training_quantization/torch_fx/resnet18/README.md), [TensorFlow](/examples/post_training_quantization/tensorflow/mobilenet_v2/README.md), [ONNX](/examples/post_training_quantization/onnx/mobilenet_v2/README.md), and [OpenVINO](/examples/post_training_quantization/openvino/mobilenet_v2/README.md)

In case the Post-Training Quantization algorithm could not reach quality requirements you can fine-tune a quantized pytorch model. Example of the Quantization-Aware training pipeline for a pytorch model could be found [here](/examples/quantization_aware_training/torch/resnet18/README.md).

## Using `pytorch.Dataloader` or `tf.data.Dataset` as data source for calibration dataset

```batch_size``` is a parameter of a dataloader that refers to the number of samples or data points propagated through the neural network in a single pass.

NNCF allows for dataloaders with different batch sizes, but there are limitations. For models like transformers or those with unconventional tensor structures, such as the batch axis not being in the expected position, using batch sizes larger than 1 for quantization isn't supported. It happens because certain models' internal data arrangements may not align with the assumptions made during quantization, leading to inaccurate statistics calculation issues with batch sizes larger than 1.

Please keep in mind that you have to recalculate the subset size for quantization according to the batch size using the following formula: ```subset_size = subset_size_for_batch_size_1 // batch_size.```.

[Example](/examples/post_training_quantization/torch/mobilenet_v2/README.md) with post-training quantization for PyTorch with a dataloader having a ```batch_size``` of 128.
