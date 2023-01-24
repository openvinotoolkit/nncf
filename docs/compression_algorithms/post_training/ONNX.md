## Post-Training Quantization for ONNX

NNCF supports [ONNX](https://onnx.ai/) backend for the Post-Training Quantization algorithm.
This guide contains some notes that you should consider before working with NNCF for ONNX.

### Model Preparation

The majority of the ONNX models are exported from different frameworks, such as PyTorch or TensorFlow.

NNCF fully supports ONNX models with the opset 13 or higher. \
NNCF supports only per-tensor quantization for ONNX models with the opset 10, 11, and 12. \
NNCF does not support ONNX models with the opset lower than 10.

If you have an ONNX model with the opset lower than 13, we recommend to update the model to a higher opset version. \
If you obtained an ONNX model from other frameworks, you can try to update an export function by setting a higher target opset version. \
If you do not have access to the export function, you can use the converter function from the ONNX package. See the example below.

```python
import onnx
from onnx.version_converter import convert_version

model = onnx.load_model('/path_to_model')
converted_model = convert_version(model, target_version=13)
```