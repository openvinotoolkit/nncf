# Post-Training Quantization of MobileNet v2 OpenVINO Model
This example demonstrates how to use Post-Training Quantization API from Neural Network Compression Framework (NNCF) to quantize YOLOv8n model.


The example includes the following steps:
- Download and prepare COCO-128 dataset.
- Quantize the model with NNCF Post-Training Quantization algorithm.
- Measure accuracy and performance of the floating-point and quantized models.

# Install requirements
At this point it is assumed that you have already installed NNCF. You can find information on installation NNCF [here](https://github.com/openvinotoolkit/nncf#user-content-installation).

To work with the example you should install the corresponding Python package dependencies:
```
pip install -r requirements.txt
```

# Run Example
The example is fully automated. Just run the following comman in the prepared Python environment:
```
python main.py
```

## See also
- [YOLOv8 Jupyter notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/230-yolov8-optimization)