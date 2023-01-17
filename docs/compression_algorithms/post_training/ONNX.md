## Post-Training Quantization for ONNX

ONNX is one of the supported backend for the Post-Training Quantization algorithm.
This guide contains some notes that you have to consider working with ONNX.

### Model preparation

The most of the ONNX models are exported from different frameworks such as PyTorch or Tensorflow.
ONNX supports the per-channel quantizaiton layers starting with opset 13.
If you have the models with the lower opset we recommend you to change the export to get the opset higher than 13 or use
the native converter function from onnx package.

```python
import onnx
from onnx.version_converter import convert_version

model = onnx.load_model('/path_to_model')
converted_model = convert_version(model, target_version=13)
```