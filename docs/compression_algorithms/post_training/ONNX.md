## Post-Training Quantization for ONNX

ONNX is one of the supported backend for the Post-Training Quantization algorithm.
This guide contains some notes that you have to consider working with NNCF for ONNX.

### Model preparation

The most of the ONNX models are exported from different frameworks such as PyTorch or Tensorflow.

NNCF fully supports ONNX models starting with opset 13. \
NNCF supports only per-tensor quantization for ONNX models having opset 10 or higher but lower 13. \
NNCF does not support ONNX models with opset lower than 10.

If you have the ONNX model with the opset lower than 13 we recommend you to update the model's opset version. The recommended version is 13 opset. \
If you export ONNX model from other framework you, probably, can update an export function from framework to ONNX by setting a new target opset version. \
If you do not have the access to export function you can use the native converter function from onnx package.

```python
import onnx
from onnx.version_converter import convert_version

model = onnx.load_model('/path_to_model')
converted_model = convert_version(model, target_version=13)
```