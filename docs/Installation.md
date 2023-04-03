## Installation
We suggest to install or use the package in the [Python virtual environment](https://docs.python.org/3/tutorial/venv.html).

If you want to optimize a model from PyTorch, install PyTorch by following [PyTorch installation guide](https://pytorch.org/get-started/locally/#start-locally). 
If you want to optimize a model from TensorFlow, install TensorFlow by following [TensorFlow installation guide](https://www.tensorflow.org/install/).

#### As a package built from a checked-out repository:

Install the package and its dependencies by running the following in the repository root directory:
```
pip install .
```

Note that if you install NNCF in this manner, the backend frameworks supported by NNCF will not be explicitly installed. NNCF will try to work with whatever backend versions you have installed in your Python environment.

If you want to install both NNCF and the supported PyTorch version in one line, you can do this by running:
```
pip install .[torch]
```
For installation of NNCF along with TensorFlow, run:
```
pip install .[tf]
```
For installation of NNCF for ONNX, run:
```
pip install .[onnx]
```
(Preview) For installation of NNCF for OpenVINO, run:
```
pip install .[openvino]
```


_NB_: For launching example scripts in this repository, we recommend setting the `PYTHONPATH` variable to the root of the checked-out repository once the installation is completed.

#### As a PyPI package:

NNCF can be installed as a regular PyPI package via pip:
```
pip install nncf
```
Use the same `pip install` syntax as above to install NNCF along with the backend package versions in one go, i.e. for NNCF with PyTorch, run:
```
pip install nncf[torch]
```
For installation of NNCF along with TensorFlow, run:
```
pip install nncf[tf]
```
For installation of NNCF for ONNX, run:
```
pip install nncf[onnx]
```
(Preview) For installation of NNCF for OpenVINO, run:
```
pip install nncf[openvino]
```

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
