
# Object detection sample

This sample shows an example of quantization of YOLOV5-family models from [ultralytics repository](https://github.com/ultralytics/yolov5). 
The used dataset is COCO downloaded with [ultralytics repository](https://github.com/ultralytics/yolov5).

## Install 

To correctly use the sample you should follow the instructions below.

0. Install requirements

    ```
    pip install -r <nncf dir>/nncf/experimental/onnx/requirements.txt
    ```

1. Clone yolov5 repository into object_detection sample directory

    ```
    cd <nncf dir>/examples/experimental/post_training_quantization_onnx/object_detection
    git clone https://github.com/ultralytics/yolov5  
    cd yolov5  
    pip install -r requirements.txt
    cd ..
    ```

2. Download dataset

    ```
    cd <nncf dir>/examples/experimental/post_training_quantization_onnx/object_detection/yolov5
    bash data/scripts/get_coco.sh
    ```

Now, use the 
```<nncf dir>/examples/experimental/post_training_quantization_onnx/object_detection/datasets/coco``` as the path to data.

## Get ONNX Yolov5 model

Take a notice that the ONNX model opset should be equal 13.

```
cd yolov5
python export.py --weights yolov5s.pt --opset 13
```

## Run Post-Training quantization sample

Important: There are several operations in the last layers of the model that should be considered as non-quantizable.
Quantization of these layers drops the accuracy significantly. It was figured out empirically. Please, add them to --ignore_scopes parameter.
To identify these layers we recommend to open the ONNX model via [Netron](https://github.com/lutzroeder/netron) and check out the layers name, which are presented on the image.



These are layers Mul and Add operations at the end of the network, which are inside red rectangles. The necessary layer name could be found in Properties.

  

![This is an image](./yolov5_last_layers.jpg)


```
python post_training_quantization.py -m <ONNX model path> -o <quantized ONNX model path> --data <COCO data path> --init_samples 100 --ignored_scopes Mul_222 Add_226 Mul_228 Mul_276 Add_280 Mul_282 Mul_235 Mul_239 Mul_289 Mul_293 Mul_330 Add_334 Mul_336 Mul_343 Mul_347
```

## Run validation

You could validate the accuracy of INT8 model by running the following command:

```
python val.py --data data/coco.yaml --weights <quantized ONNX model path> 
```


## Results of Post-Training quantization of ONNX model

|Model|Original accuracy|Quantized model accuracy|
| :---: | :---: | :---: |
|Yolov5s|37.1%|36.1%|