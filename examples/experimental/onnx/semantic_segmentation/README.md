# Semantic Segmentation example

This example demonstrates quantization of UNet and ICNet segmentation models on CamVid and Mapillary datasets.

## Install

Please, install the requirements for ONNX Post-Training Quantization of NNCF.

Install requirements

```
pip install -r <nncf dir>/nncf/experimental/onnx/requirements.txt
pip install -r <nncf dir>/examples/experimental/onnx/requirements.txt
```

## Quantizing the model

To run post-training quantization on your model you can use the following command.

```
python onnx_ptq_segmentation.py -m <ONNX model path> -o <quantized ONNX model path> --dataset_name <CamVid or Mapillary> --data <dataset path>
```

Also, you could specify some options of quantization, please, take a look at the argument description by using the
command:

```
python onnx_ptq_segmentation.py --help
```

## Validating the accuracy of the original and quantized models

If you would like to compare the accuracy of the original model and quantized one, you could
use [accuracy_checker](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/accuracy_checker). The
necessary config files are located [here](./examples/experimental/onnx/semantic_segmentation/ac_configs/). The thing that you only need is to
fill in the config with the following infromation: the path to ImageNet folder and the path to the annotation file. The
accuracy checker config for the original and quantized models is the same.

Use the following command to get the model accuracy:

```
accuracy_check -c <path to config fileh> -m <ONNX model>
```

## Results of Post-Training quantization of ONNX models

| Model |  Original accuracy  | Quantized model accuracy |
|:-----:|:-------------------:|:------------------------:|
| ICNet |   67.88% (CamVid)   |      67.8% (CamVid)      |
| UNet  |   71.95% (CamVid)   |     71.85% (CamVid)      |
| UNet  |  56.24% (Mapillary) |     56.19% (Mapillary)   |

## Benchmarking the original and quantized models

If you would like to compare the performance of the original model and quantized one, you could
use [benchmark_tool](https://github.com/openvinotoolkit/openvino/tree/master/tools/benchmark_tool).

Use the following command to get the model performance numbers:

```
benchmark_app -m <ONNX model>
```
