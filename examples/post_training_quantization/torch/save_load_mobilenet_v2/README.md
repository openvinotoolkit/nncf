## Post-Training Quantization Saving and Loading for MobileNet v2 PyTorch Model

This example demonstrates how to use Post-Training Quantization API from Neural Network Compression Framework (NNCF) to quantize, save, load and export PyTorch models on the example of [MobileNet v2](https://huggingface.co/alexsu52/mobilenet_v2_imagenette) quantization, pretrained on [Imagenette](https://github.com/fastai/imagenette) dataset.

The example includes three files:

- quantize.py
  - Loading the [Imagenette](https://github.com/fastai/imagenette) dataset (~340 Mb) and the [MobileNet v2 PyTorch model](https://huggingface.co/alexsu52/mobilenet_v2_imagenette) pretrained on this dataset.
  - Quantizing the model using NNCF Post-Training Quantization algorithm.
  - Saving the quantized PyTorch model checkpoint to a binary file.

- export_openvino.py
  - Recovering the saved quantized PyTorch model from the checkpoint.
  - Converting of the original and the quantized PyTorch models to Openvino IR.
  - Checking performance, accuracy and size of both original and quantized models.

- export_onnx.py
  - Recovering the saved quantized PyTorch model from the checkpoint.
  - Converting of the original and the quantized PyTorch models to onnx format.
  - Checking performance, accuracy and size of both original and quantized models.

## Install requirements

At this point it is assumed that you have already installed NNCF. You can find information on installation NNCF [here](https://github.com/openvinotoolkit/nncf#user-content-installation).

To work with the example you should install the corresponding Python package dependencies:

```bash
pip install -r requirements.txt
```

## Run Example

It's pretty simple. The example does not require additional preparation. It will do the preparation itself, such as loading the dataset and model, etc.

```bash
python quantize.py
```

will produce a bin file with information required for the quantized PyTorch model recovering. To export and validate models run

```bash
python export_openvino.py
```

to use Openvino IR format as the target format or

```bash
python export_onnx.py
```

to use Openvino ONNX format as the target format otherwise.
