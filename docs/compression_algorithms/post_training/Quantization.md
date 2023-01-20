

## Post-Training Quantization

Post-Training Quantization is the quantization algorithm which doesn't demand retraining of a quantized model. 
It utilizes a small subset of the initial dataset to calibrate quantization constants.

NNCF provides an advanced Post-Training Quantization algorithm, which consists of following techniques:

1) MinMaxQuantization - Analyzes model and inserts extra quantization layers
with parameters are calibrated using the small subset. 
2) FastBiasCorrection or BiasCorrection - Reduces the bias errors between the quantized layers and the corresponding original layers.


### Usage

To start the algorithm the user should provide:
* Original model.
* Validation part of the dataset.
* [Data transformation function](#data-transfomation-function) from original dataset format to the NNCF format.


The basic workflow steps:
1) Create the [data transformation function](#data-transfomation-function).
2) Initialize NNCF Dataset with the validation dataset and the transformation function.
3) Run the quantization pipeline.

#### Data Transformation function

Every user training pipeline consumes data in the unique format to feed the model.
These data formats differ from pipeline to pipeline, thus NNCF introduces the data transformation function - to provide the interface to adapt the user dataset format to the NNCF format.

Every backend demands own return value format for transformation function, which is based on the input format of the backend inference framework.
Below there are formats of transformation function for each suported backend.

<details><summary><b>PyTorch, TensorFlow, OpenVINO</b></summary>
The return format of data transformation function is directly the input tensors, consumed by the model.

If you are not sure that your implementation of data transformation function is correct you can validate it by using the following code:
```python
model = ... # Model
val_loader = ... # Original Dataset
transform_fn = ... # Data transformation function
for data_item in val_loader:
    model(transform_fn(data_item))
```


</details>
<details><summary><b>ONNX</b></summary>

[ONNXRuntime](https://onnxruntime.ai/) is used as the inference engine for ONNX. \
The input format of the data which is used by ONNXRuntime is following - ```Dict[str, np.ndarray]```, where the keys of the dict are names of the model inputs and the values are the numpy tensors passed to these inputs.

If you are not sure that your implementation of data transformation function is correct you can validate it by using the following code:
```python
import onnxruntime
model_path = ... # Path to Model
val_loader = ... # Original Dataset
transform_fn = ... # Data transformation function
sess = onnxruntime.InferenceSession(model_path)
output_names = [output.name for output in sess.get_outputs()]
for data_item in val_loader:
    sess.run(output_names, input_feed=transform_fn(data_item))
```

</details>

NNCF provides the examples of Post-Training Quantization where you can find the implementation of data transformation function: [PyTorch](../../../examples/post_training_quantization/torch/mobilenet_v2/README.md), [TensorFlow](../../../examples/post_training_quantization/tensorflow/mobilenet_v2/README.md), [ONNX](../../../examples/post_training_quantization/onnx/mobilenet_v2/README.md), [OpenVINO](../../../examples/post_training_quantization/openvino/mobilenet_v2/README.md)