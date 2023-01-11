PYLINT_VERSION := 2.13.9
JUNITXML_PATH ?= nncf-tests.xml
MODELS_DIR ?= /models
DATA_DIR ?= /datasets
OUTPUT_DIR ?= /output
ANNO_DIR ?= /annots

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
	pytest tests/onnx --junitxml ${JUNITXML_PATH}

ONNX_PYFILES := $(shell find examples/post_training_quantization/onnx -type f -name "*.py")
pylint-onnx:
	pylint --rcfile .pylintrc               \
		nncf/experimental/onnx              \
		nncf/quantization                   \
		tests/onnx                          \
		$(ONNX_PYFILES)

test-install-onnx:
	pytest tests/cross_fw/install/ -s       \
		--backend onnx                      \
		--junitxml ${JUNITXML_PATH}

ONNX_E2E_PTQ_SIZE ?= 300
ONNX_E2E_PTQ_EVAL_SIZE ?=
ONNX_E2E_OPTIONS=--model-dir ${MODELS_DIR} --data-dir ${DATA_DIR}  \
				 --output-dir ${OUTPUT_DIR} --anno-dir ${ANNO_DIR} \
                 --junitxml ${JUNITXML_PATH} --ptq-size ${ONNX_E2E_PTQ_SIZE} \
                 --eval-size ${ONNX_E2E_PTQ_EVAL_SIZE}

test-e2e-ptq:
	pytest tests/onnx -m e2e_ptq ${ONNX_E2E_OPTIONS} --enable-cpu-ep --disable-ov-ep



###############################################################################
# OpenVino backend
install-openvino-test:
	pip install -U pip
	pip install -e .[openvino]
	pip install -r tests/openvino/requirements.txt
	pip install -r tests/cross_fw/install/requirements.txt
	pip install -r examples/experimental/openvino/bert/requirements.txt
	pip install -r examples/experimental/openvino/yolo_v5/requirements.txt

install-openvino-dev: install-openvino-test
	pip install pylint==$(PYLINT_VERSION)

test-openvino:
	pytest tests/openvino --junitxml ${JUNITXML_PATH}

pylint-openvino:
	pylint --rcfile .pylintrc               \
		nncf/openvino/                      \
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
