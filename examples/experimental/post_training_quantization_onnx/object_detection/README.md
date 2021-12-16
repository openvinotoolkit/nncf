
# Object detection sample

This sample shows an example of quantization of YOLOV5-family models from [ultralytics repository](https://github.com/ultralytics/yolov5). 
The used dataset is COCO downloaded with [ultralytics repository](https://github.com/ultralytics/yolov5).

## Install 

To correctly use the sample you should follow the instructions below.

1. Clone yolov5 repositry

```
git clone https://github.com/ultralytics/yolov5  
cd yolov5  
pip install -r requirments.txt
```

2. Download dataset

To download dataset, please, take a look at the instructions at yolov5 repository. Spoiler: you can run train.py script which automatically download the data.

## Get ONNX Yolov5 model

Take a notice that the ONNX model opset should be equal 13.

```
cd yolov5
python export.py --weights yolov5s.pt --opset 13
```

## Run Post-Training quantization sample

Important: There are several operations in the last layers of the model that should be considered as non-quantizable.
Quantization of these layers drop the accuracy significantly. Please, add them to --ignore_scopes parameter.

```
python post_training_quantization.py -m <ONNX model path> -o <quantized ONNX model path> --data <COCO data path> --init_samples 100 --ignored_scopes Mul_222 Add_226 Mul_228 Mul_276 Add_280 Mul_282 Mul_235 Mul_239 Mul_289 Mul_293 Mul_330 Add_334 Mul_336 Mul_343 Mul_347
```

## Results of Post-Training quantization of ONNX model

|Model|Original accuracy|Quantized model accuracy|
| :---: | :---: | :---: |
|Yolov5s|37.1%|36.1%|