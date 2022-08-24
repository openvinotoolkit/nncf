# Support ONNXRuntime PTQ for Yolov5 models

## Prerequisite

1. Follow the [installation step](#installation)
2. Clone [Yolov5 repository](https://github.com/ultralytics/yolov5.git) and patch it.

```bash
$ cd examples/experimental/onnx/yolov5
$ git init
$ git remote add origin https://github.com/ultralytics/yolov5.git
$ git fetch origin
$ git checkout 34df5032a7d2e83fe3d16770a03bd129b115d184
$ git apply 0001-Add-NNCF-ONNX-PTQ-example-notebook.patch
```

## Run NNCF ONNXRuntime PTQ

After [prerequisite](#prerequisite) is done, you can find `run_notebook.ipynb` notebook file in `examples/experimental/onnx/yolov5`. If you finish running all notebook cells, you will obtain the following PTQ benchmark results.

```
# Model accuracy
FP32 mAP: 37.1%, INT8 mAP: 36.4%, mAP difference: 0.8%
```
