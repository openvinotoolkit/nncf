JUNITXML_PATH ?= nncf-tests.xml

ifdef DATA
	DATA_ARG := --data $(DATA)
endif

install-pre-commit:
	pip install pre-commit==3.2.2

install-pylint:
	pip install pylint==2.13.9
	pip install pylintfileheader==0.3.2

###############################################################################
# ONNX backend
install-onnx-test:
	pip install -U pip
	pip install -e .[onnx]
	pip install -r tests/onnx/requirements.txt
	pip install -r tests/cross_fw/install/requirements.txt
	pip install -r tests/onnx/benchmarking/requirements.txt
	pip install -r examples/post_training_quantization/onnx/mobilenet_v2/requirements.txt

install-onnx-dev: install-onnx-test install-pre-commit install-pylint

test-onnx:
	pytest tests/onnx $(DATA_ARG) --junitxml ${JUNITXML_PATH}

ONNX_PYFILES := $(shell git ls-files | grep -P ".*onnx.*py$$" | grep -v -P ".*torch.*" | grep -v -P "^tools" )
pylint-onnx:
	pylint --rcfile .pylintrc               \
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

install-openvino-dev: install-openvino-test install-pre-commit install-pylint
	pip install -r examples/post_training_quantization/openvino/mobilenet_v2/requirements.txt
	pip install -r examples/post_training_quantization/openvino/quantize_with_accuracy_control/requirements.txt
	pip install -r examples/post_training_quantization/openvino/yolov8/requirements.txt
	pip install -r examples/post_training_quantization/openvino/yolov8_quantize_with_accuracy_control/requirements.txt

test-openvino:
	pytest tests/openvino $(DATA_ARG) --junitxml ${JUNITXML_PATH}

OV_PYFILES := $(shell git ls-files | grep -P ".*openvino.*py$$" | grep -v -P "^tools")
pylint-openvino:
	pylint --rcfile .pylintrc               \
		$(OV_PYFILES)

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

install-tensorflow-dev: install-tensorflow-test install-pre-commit install-pylint

test-tensorflow:
	pytest tests/common tests/tensorflow    \
		--junitxml ${JUNITXML_PATH}         \
		$(DATA_ARG)

TF_PYFILES := $(shell git ls-files | grep -P ".*tensorflow.*py$$" | grep -v -P "^tools")
pylint-tensorflow:
	pylint --rcfile .pylintrc               \
		$(TF_PYFILES)

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

install-torch-dev: install-torch-test install-pre-commit install-pylint
	pip install -r examples/post_training_quantization/torch/ssd300_vgg16/requirements.txt
	pip install -r docs/api/requirements.txt

test-torch:
	pytest tests/common tests/torch --junitxml ${JUNITXML_PATH} $(DATA_ARG)

TORCH_PYFILES := $(shell git ls-files | grep -P ".*torch.*py$$" | grep -v -P "^tools")
COMMON_PYFILES := $(shell git ls-files | grep -P ".*py$$" \
					| grep -v -P ".*(torch|tensorflow|onnx|openvino).*" \
					| grep -v -P "^tools")

pylint-torch:
	pylint --rcfile .pylintrc   \
		$(TORCH_PYFILES)        \
		$(COMMON_PYFILES)


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
	pre-commit run -a
