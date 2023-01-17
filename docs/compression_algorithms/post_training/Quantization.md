

## Post-Training Quantization

Post-Training Quantization is the quantization algorithm which doesn't demand training of a model. 
It utilizes a small subset of the initial dataset to calibrate quantization constants.

NNCF provides an advanced Post-Training Quantization algorithm, which consists of following techniques:

1) MinMaxQuantization - Analyzes model and insert extra quantization layers
with parameters are calibrated using the small subset. 
2) FastBiasCorrection or BiasCorrection - Reduces the bias errors of the quantized layers and original layers.


### Usage

To start the algorithm the user should provide:
1) Original model.
2) Validation part of the dataset.
3) Data transformation function. The user training pipeline needs data in the own format to feed it to the model. 
These formats differ from pipeline to pipeline, thus NNCF introduces the data transformation function - 
to provide the interface from the user dataset format to NNCF format. \
Every backend demands its own return value format of transformation function, which is based on the input of the inference framework used.

The basic workflow steps:
1) Initialize the data transformation function.
2) Initialize NNCF Dataset with the validation dataset.
3) Run the quantization pipeline.


#### ONNX

[ONNXRuntime](https://onnxruntime.ai/) is used as the inference engine for ONNX backend. \
The input format of the data which is used by ONNXRuntime is following - Dict[str, np.ndarray], where the keys of the dict are string names of the model inputs and the values are the tensors consumed by these inputs.
So, the transformation function should return the data in this format.

Let's take a look at the usage example.

```python
import onnx
import nncf
import torch
from torchvision import datasets

# Instantiate your uncompressed model
onnx_model = onnx.load_model('/model_path')
# Provide validation part of the dataset for statistics collection for compression algorithm
representative_dataset = datasets.ImageFolder("/path")
dataset_loader = torch.utils.data.DataLoader(representative_dataset, batch_size=1)
# Step 1: Initialize transformation function
input_name = onnx_model.graph.input[0].name
def transform_fn(data_item):
    images, _ = data_item
    return {input_name: images.numpy()}
# Step 2: Initialize NNCF Dataset
calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
# Step 3: Run the quantization pipeline
quantized_model = nncf.quantize(onnx_model, calibration_dataset)
```

