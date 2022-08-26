# Semantic Segmentation example

This example demonstrates quantization of UNet and ICNet segmentation models on CamVid and Mapillary datasets.

## Install

Please, install the requirements for ONNX Post-Training Quantization of NNCF.

Install requirements

```
pip install -r <nncf dir>/nncf/experimental/onnx/requirements.txt
pip install -r <nncf dir>/examples/experimental/onnx/requirements.txt
```

## Getting the quantized model

To run post-training quantization on your model you can use the following command.

```
python onnx_ptq_segmentation.py -m <ONNX model path> -o <quantized ONNX model path> --dataset_name <CamVid or Mapillary> --data <dataset path>
```

Also, you could specify some options of quantization, please, take a look at the argument description by using the
command:

```
python onnx_ptq_segmentation.py --help
```

## Results of Post-Training quantization of ONNX models

| Model |  Original accuracy  | Quantized model accuracy |
|:-----:|:-------------------:|:------------------------:|
| ICNet |   67.88% (CamVid)   |      67.8% (CamVid)      |
| UNet  |   71.95% (CamVid)   |     71.85% (CamVid)      |
| UNet  |  56.24% (Mapillary) |     56.19% (Mapillary)   |
