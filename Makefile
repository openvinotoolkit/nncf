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

NNCF_DIR ?=
create_onnx_ptq_e2e_venv:
	pip install -U pip
	pip install -e ${NNCF_DIR}[onnx]
	pip install -r ${NNCF_DIR}/tests/onnx/requirements.txt
	pip install pycocotools
	pip install cython
	yes | pip uninstall onnxruntime-openvino
	git lfs pull --include ${NNCF_DIR}tests/onnx/onnxruntime_openvino-1.14.0-cp38-cp38-linux_x86_64.whl
	git lfs pull --include ${NNCF_DIR}tests/onnx/openvino-2022.3.0-8784-cp38-cp38-manylinux_2_31_x86_64.whl
	git lfs pull --include ${NNCF_DIR}tests/onnx/openvino_dev-2022.3.0-8784-py3-none-any.whl
	pip install ${NNCF_DIR}/tests/onnx/openvino-2022.3.0-8784-cp38-cp38-manylinux_2_31_x86_64.whl
	pip install ${NNCF_DIR}/tests/onnx/openvino_dev-2022.3.0-8784-py3-none-any.whl
	pip install ${NNCF_DIR}/tests/onnx/onnxruntime_openvino-1.14.0-cp38-cp38-linux_x86_64.whl
	pip install numpy==1.23.1
