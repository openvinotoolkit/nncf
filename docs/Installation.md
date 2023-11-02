# Installation

We suggest to install or use the package in the [Python virtual environment](https://docs.python.org/3/tutorial/venv.html).

If you want to optimize a model from PyTorch, install PyTorch by following [PyTorch installation guide](https://pytorch.org/get-started/locally/#start-locally). For other backend follow: [TensorFlow installation guide](https://www.tensorflow.org/install/), [ONNX installation guide](https://onnxruntime.ai/docs/install/), [OpenVINO installation guide](https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html).

## As a PyPI package

NNCF can be installed as a regular PyPI package via pip:

```bash
pip install nncf
```

If you want to install both NNCF and the supported PyTorch version in one line, you can do this by simply running:

```bash
pip install nncf[torch]
```

Other viable options besides `[torch]` are `[tf]`, `[onnx]` and `[openvino]`.

## As a package built from a checked-out repository

Install the package and its dependencies by running the following command in the repository root directory:

```bash
pip install .
```

Use the same `pip install` syntax as above to install NNCF along with the backend package version in one go:

```bash
pip install .[<BACKEND>]
```

List of supported backends: `torch`, `tf`, `onnx` and `openvino`.

For development purposes install extra packages by

```bash
pip install .[dev,tests]
```

_NB_: For launching example scripts in this repository, we recommend setting the `PYTHONPATH` variable to the root of the checked-out repository once the installation is completed.

NNCF is also available via [conda](https://anaconda.org/conda-forge/nncf):

```bash
conda install -c conda-forge nncf
```

## From a specific commit hash using pip

```python
pip install git+https://github.com/openvinotoolkit/nncf@bd189e2#egg=nncf
```

Note that in order for this to work for pip versions >= 21.3, your Git version must be at least 2.22.

## As a Docker image

Use one of the Dockerfiles in the [docker](../docker) directory to build an image with an environment already set up and ready for running NNCF [sample scripts](../README.md#model-compression-samples).

## Corresponding versions

The following table lists the recommended corresponding versions of backend packages
as well as the supported versions of Python:

| NNCF      | OpenVINO   | PyTorch  | ONNX     | TensorFlow | Python |
|-----------|------------|----------|----------|------------|--------|
| `develop` | `2023.1.0` | `2.1`    | `1.13.1` | `2.12.0`   | `3.8`  |
| `2.6.0`   | `2023.1.0` | `2.0.1`  | `1.13.1` | `2.12.0`   | `3.8`  |
| `2.5.0`   | `2023.0.0` | `1.13.1` | `1.13.1` | `2.11.1`   | `3.8`  |
| `2.4.0`   | `2022.1.0` | `1.12.1` | `1.12.0` | `2.8.2`    | `3.8`  |
