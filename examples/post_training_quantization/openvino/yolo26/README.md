# Post-Training Quantization of YOLO26 OpenVINO Model

This example demonstrates how to use Post-Training Quantization API from Neural Network Compression Framework (NNCF) to quantize YOLO26 model.

The example includes the following steps:

- Download and prepare COCO-128 dataset.
- Quantize the model with NNCF Post-Training Quantization algorithm.
- Measure accuracy and performance of the floating-point and quantized models.

## Prerequisites

Before running this example, ensure you have Python 3.10+ installed and set up your environment:

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
