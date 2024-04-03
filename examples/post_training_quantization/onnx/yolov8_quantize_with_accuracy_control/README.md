# Post-Training Quantization of YOLOv8 ONNX Model

This example demonstrates how to use Post-Training Quantization API from Neural Network Compression Framework (NNCF) to quantize YOLOv8n ONNX model
with accuracy control.

The example includes the following steps:

- Download and prepare COCO-128-seg dataset.
- Quantize the model with accuracy control.
- Measure accuracy and performance of the floating-point and quantized models.
- Convert the resulted INT8 model to the OpenVINO format.

## Install requirements

To run the example you should install the corresponding Python dependencies:

- Install NNCF from source:

    ```bash
    git clone https://github.com/openvinotoolkit/nncf.git
    cd nncf
    pip install .
    ```

- Install 3rd party dependencies of this example:

    ```bash
    pip install -r requirements.txt
    ```

## Run Example

The example is fully automated. Just run the following command in the prepared Python environment:

```bash
python main.py
```

Run the following command to convert resulted INT8 model to the OpenVINO format:

```bash
python deploy.py
```

## See also

- [YOLOv8 Jupyter notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov8-optimization)
