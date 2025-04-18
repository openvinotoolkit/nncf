# Post-Training Quantization of YOLOv8 OpenVINO Model

This example demonstrates how to use Post-Training Quantization API from Neural Network Compression Framework (NNCF) to quantize YOLOv8n model.

The example includes the following steps:

- Download and prepare COCO-128 dataset.
- Quantize the model with NNCF Post-Training Quantization algorithm.
- Measure accuracy and performance of the floating-point and quantized models.

## Install requirements

To run the example you should install the corresponding Python dependencies:

- Install NNCF from source:

```bash
pip install ../../../../
```

- Install 3rd party dependencies:

```bash
pip install -r requirements.txt
```

## Run Example

The example is fully automated. Just run the following command in the prepared Python environment:

```bash
python main.py
```

## See also

- [YOLOv8 Jupyter notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov8-optimization)
