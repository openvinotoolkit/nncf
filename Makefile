PYLINT_VERSION := 2.13.9
JUNITXML_PATH ?= nncf-tests.xml

ifdef DATA
	DATA_ARG := --data $(DATA)
endif

###############################################################################
# ONNX backend
install-onnx-test:
	pip install -U pip
	pip install -e .[onnx]
	pip install -r tests/onnx/requirements.txt
	pip install -r tests/cross_fw/install/requirements.txt
	pip install -r tests/onnx/benchmarking/requirements.txt
	pip install -r examples/post_training_quantization/onnx/mobilenet_v2/requirements.txt

install-onnx-dev: install-onnx-test
	pip install pylint==$(PYLINT_VERSION)

test-onnx:
	pytest tests/onnx $(DATA_ARG) --junitxml ${JUNITXML_PATH}

ONNX_PYFILES := $(shell find examples/post_training_quantization/onnx -type f -name "*.py")
pylint-onnx:
	pylint --rcfile .pylintrc               \
		nncf/onnx              	            \
		nncf/quantization                   \
		tests/onnx                          \
		$(ONNX_PYFILES)

test-install-onnx:
	pytest tests/cross_fw/install/ -s       \
		--backend onnx                      \
		--junitxml ${JUNITXML_PATH}

###############################################################################
# OpenVino backend
install-openvino-test:
	pip install -U pip
	pip install -e .[openvino]
	pip install -r tests/openvino/requirements.txt
	pip install -r tests/cross_fw/install/requirements.txt
	pip install -r examples/experimental/openvino/bert/requirements.txt
	pip install -r examples/experimental/openvino/yolo_v5/requirements.txt
	pip install git+https://github.com/openvinotoolkit/open_model_zoo.git#subdirectory=tools/model_tools

install-openvino-dev: install-openvino-test
	pip install pylint==$(PYLINT_VERSION)

test-openvino:
	pytest tests/openvino $(DATA_ARG) --junitxml ${JUNITXML_PATH}

pylint-openvino:
	pylint --rcfile .pylintrc               \
		nncf/openvino/                      \
		nncf/experimental/openvino/  \
		tests/openvino/                     \
		examples/experimental/openvino/

test-install-openvino:
	pytest tests/cross_fw/install -s        \
		--backend openvino                  \
		--junitxml ${JUNITXML_PATH}

###############################################################################
# TensorFlow backend
install-tensorflow-test:
	pip install -U pip
	pip install -e .[tf]
	pip install -r tests/tensorflow/requirements.txt
	pip install -r tests/cross_fw/install/requirements.txt
	pip install -r examples/tensorflow/requirements.txt

install-tensorflow-dev: install-tensorflow-test
	pip install pylint==$(PYLINT_VERSION)

test-tensorflow:
	pytest tests/common tests/tensorflow    \
		--junitxml ${JUNITXML_PATH}         \
		$(DATA_ARG)

pylint-tensorflow:
	pylint --rcfile .pylintrc               \
		nncf/tensorflow                     \
		nncf/experimental/tensorflow        \
		tests/tensorflow                    \
		tests/experimental/tensorflow       \
		examples/tensorflow

test-install-tensorflow:
	pytest tests/cross_fw/install/ -s --backend tf --junitxml ${JUNITXML_PATH}

###############################################################################
# PyTorch backend
install-torch-test:
	pip install -U pip
	pip install -e .[torch]
	pip install -r tests/torch/requirements.txt
	pip install -r tests/cross_fw/install/requirements.txt
	pip install -r examples/torch/requirements.txt

install-torch-dev: install-torch-test
	pip install pylint==$(PYLINT_VERSION)

test-torch:
	pytest tests/common tests/torch --junitxml ${JUNITXML_PATH} $(DATA_ARG)

pylint-torch:
	pylint --rcfile .pylintrc               \
		nncf/common                         \
		nncf/config                         \
		nncf/api                            \
		nncf/torch                          \
		nncf/experimental/torch             \
		tests/common                        \
		tests/torch                         \
		examples/torch                      \
		examples/experimental/torch

test-install-torch-cpu:
	pytest tests/cross_fw/install/ -s       \
		--backend torch                     \
		--host-configuration cpu            \
		--junitxml ${JUNITXML_PATH}

test-install-torch-gpu:
	pytest tests/cross_fw/install -s        \
		--backend torch                     \
		--junitxml ${JUNITXML_PATH}

###############################################################################
# Pre commit check
pre-commit:
	pip install pre-commit==3.2.2
	pre-commit run -a
