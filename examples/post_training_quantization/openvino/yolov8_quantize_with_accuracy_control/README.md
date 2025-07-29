# Post-Training Quantization of YOLOv8 OpenVINO Model

This example demonstrates how to use Post-Training Quantization API from Neural Network Compression Framework (NNCF) to quantize YOLOv8n model.

The example includes the following steps:

- Download and prepare COCO-128-seg dataset.
- Quantize the model with "AccuracyAwareQuantization" algorithm instead of "DefaultQuantization".
- Measure accuracy and performance of the floating-point and quantized models.

## Prerequisites

Before running this example, ensure you have Python 3.9+ installed and set up your environment:

### 1. Create and activate a virtual environment

```bash
python3 -m venv nncf_env
source nncf_env/bin/activate  # On Windows: nncf_env\Scripts\activate.bat
```

### 2. Install NNCF and other dependencies

```bash
python3 -m pip install ../../../../ -r requirements.txt
```

## Run Example

The example is fully automated. Just run the following command in the prepared Python environment:

```bash
python main.py
```

## See also

- [YOLOv8 Jupyter notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov8-optimization)
