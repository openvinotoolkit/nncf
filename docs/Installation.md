## Installation
We suggest to install or use the package in the [Python virtual environment](https://docs.python.org/3/tutorial/venv.html).

If you want to optimize a model from PyTorch, install PyTorch by following [PyTorch installation guide](https://pytorch.org/get-started/locally/#start-locally). For other backend follow: [TensorFlow installation guide](https://www.tensorflow.org/install/), [ONNX installation guide](https://onnxruntime.ai/docs/install/), [OpenVINO installation guide](https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html).

#### As a PyPI package:

NNCF can be installed as a regular PyPI package via pip:
```
pip install nncf
```
If you want to install both NNCF and the supported PyTorch version in one line, you can do this by simply running:
```
pip install nncf[torch]
```
Other viable options besides `[torch]` are `[tf]`, `[onnx]` and `[openvino]`.


#### As a package built from a checked-out repository:

Install the package and its dependencies by running the following command in the repository root directory:
```
pip install .
```

Use the same `pip install` syntax as above to install NNCF along with the backend package version in one go:
```
pip install .[<BACKEND>]
```
List of  supported backends: `torch`, `tf`, `onnx` and `openvino`.

For development purposes install extra packages by
```
pip install .[dev,tests]
```


_NB_: For launching example scripts in this repository, we recommend setting the `PYTHONPATH` variable to the root of the checked-out repository once the installation is completed.


NNCF is also available via [conda](https://anaconda.org/conda-forge/nncf):
```
conda install -c conda-forge nncf
```

#### From a specific commit hash using pip:
```python
pip install git+https://github.com/openvinotoolkit/nncf@bd189e2#egg=nncf
```
Note that in order for this to work for pip versions >= 21.3, your Git version must be at least 2.22.

#### As a Docker image
Use one of the Dockerfiles in the [docker](./docker) directory to build an image with an environment already set up and ready for running NNCF [sample scripts](#model-compression-samples).
