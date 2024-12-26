# Installation

We suggest to install or use the package in the [Python virtual environment](https://docs.python.org/3/tutorial/venv.html).

NNCF supports multiple backends. Follow the corresponding installation guides and ensure your system meets
the required specifications for your chosen backend:

- OpenVINO&trade;: [Install Guide](https://docs.openvino.ai/2024/get-started/install-openvino.html), [System Requirements](https://docs.openvino.ai/2024/about-openvino/release-notes-openvino/system-requirements.html)
- ONNX: [Install Guide](https://onnxruntime.ai/docs/install/)
- PyTorch: [Install Guide](https://pytorch.org/get-started/locally/#start-locally)
- TensorFlow: [Install Guide](https://www.tensorflow.org/install/)

## As a PyPI package

NNCF can be installed as a regular PyPI package via pip:

```bash
pip install nncf
```

## As a package built from a checked-out repository

Install the package and its dependencies by running the following command in the repository root directory:

```bash
pip install .
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

## Corresponding versions

The following table lists the recommended corresponding versions of backend packages
as well as the supported versions of Python:

| NNCF      | OpenVINO   | PyTorch  | ONNX     | TensorFlow | Python |
|-----------|------------|----------|----------|------------|--------|
| `develop` | `2024.6.0` | `2.5.1`  | `1.17.0` | `2.15.1`   | `3.10` |
| `2.14.1`  | `2024.6.0` | `2.5.1`  | `1.17.0` | `2.15.1`   | `3.10` |
| `2.14.0`  | `2024.5.0` | `2.5.1`  | `1.17.0` | `2.15.1`   | `3.10` |
| `2.13.0`  | `2024.4.0` | `2.4.0`  | `1.16.0` | `2.15.1`   | `3.8`* |
| `2.12.0`  | `2024.3.0` | `2.3.0`  | `1.16.0` | `2.15.1`   | `3.8`* |
| `2.11.0`  | `2024.2.0` | `2.3.0`  | `1.16.0` | `2.12.0`   | `3.8`  |
| `2.10.0`  | `2024.1.0` | `2.2.1`  | `1.16.0` | `2.12.0`   | `3.8`  |
| `2.9.0`   | `2024.0.0` | `2.1.2`  | `1.13.1` | `2.12.0`   | `3.8`  |
| `2.8.1`   | `2023.3.0` | `2.1.2`  | `1.13.1` | `2.12.0`   | `3.8`  |
| `2.8.0`   | `2023.3.0` | `2.1.2`  | `1.13.1` | `2.12.0`   | `3.8`  |
| `2.7.0`   | `2023.2.0` | `2.1`    | `1.13.1` | `2.12.0`   | `3.8`  |
| `2.6.0`   | `2023.1.0` | `2.0.1`  | `1.13.1` | `2.12.0`   | `3.8`  |
| `2.5.0`   | `2023.0.0` | `1.13.1` | `1.13.1` | `2.11.1`   | `3.8`  |
| `2.4.0`   | `2022.1.0` | `1.12.1` | `1.12.0` | `2.8.2`    | `3.8`  |

> (*) Python 3.9 or higher is required for TensorFlow 2.15.1

This repository is tested on Python* 3.10.14, PyTorch* 2.5.0 (NVidia CUDA\* Toolkit 12.4) and TensorFlow* 2.12.1 (NVidia CUDA\* Toolkit 11.8).
