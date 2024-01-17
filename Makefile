JUNITXML_PATH ?= nncf-tests.xml

ifdef NNCF_COVERAGE
	COVERAGE_ARGS ?= --cov=./ --cov-report=xml
else
	COVERAGE_ARGS :=
endif

ifdef DATA
	DATA_ARG := --data $(DATA)
endif

ifdef WEEKLY_MODELS
	WEEKLY_MODELS_ARG := --weekly-models $(WEEKLY_MODELS)
endif

install-pre-commit:
	pip install pre-commit==3.2.2


###############################################################################
# ONNX backend
install-onnx-test:
	pip install -U pip
	pip install -e .[onnx]
	pip install -r tests/onnx/requirements.txt
	pip install -r tests/cross_fw/install/requirements.txt
	pip install -r tests/cross_fw/examples/requirements.txt
	pip install -r tests/onnx/benchmarking/requirements.txt

install-onnx-dev: install-onnx-test install-pre-commit
	pip install -r examples/post_training_quantization/onnx/mobilenet_v2/requirements.txt

test-onnx:
	pytest ${COVERAGE_ARGS} tests/onnx $(DATA_ARG) --junitxml ${JUNITXML_PATH}


test-install-onnx:
	pytest tests/cross_fw/install -s       \
		--backend onnx                      \
		--junitxml ${JUNITXML_PATH}

test-examples-onnx:
	pytest tests/cross_fw/examples -s       \
		--backend onnx                      \
		--junitxml ${JUNITXML_PATH}

###############################################################################
# OpenVino backend
install-openvino-test:
	pip install -U pip
	pip install -e .[openvino]
	pip install tensorflow==2.12.0
	pip install -r tests/openvino/requirements.txt
	pip install -r tests/cross_fw/install/requirements.txt
	pip install -r tests/cross_fw/examples/requirements.txt

install-openvino-dev: install-openvino-test install-pre-commit
	pip install -r examples/post_training_quantization/openvino/mobilenet_v2/requirements.txt
	pip install -r examples/post_training_quantization/openvino/anomaly_stfpm_quantize_with_accuracy_control/requirements.txt
	pip install -r examples/post_training_quantization/openvino/yolov8/requirements.txt
	pip install -r examples/post_training_quantization/openvino/yolov8_quantize_with_accuracy_control/requirements.txt

test-openvino:
	ONEDNN_MAX_CPU_ISA=AVX2 pytest ${COVERAGE_ARGS} tests/openvino $(DATA_ARG) --junitxml ${JUNITXML_PATH}

test-install-openvino:
	pytest tests/cross_fw/install -s        \
		--backend openvino                  \
		--junitxml ${JUNITXML_PATH}

test-examples-openvino:
	pytest tests/cross_fw/examples -s        \
		--backend openvino                  \
		--junitxml ${JUNITXML_PATH}

###############################################################################
# TensorFlow backend
install-tensorflow-test:
	pip install -U pip
	pip install -e .[tf]
	pip install -r tests/tensorflow/requirements.txt
	pip install -r tests/cross_fw/install/requirements.txt
	pip install -r tests/cross_fw/examples/requirements.txt
	pip install -r examples/tensorflow/requirements.txt

install-tensorflow-dev: install-tensorflow-test install-pre-commit
	pip install -r examples/post_training_quantization/tensorflow/mobilenet_v2/requirements.txt

test-tensorflow:
	pytest ${COVERAGE_ARGS} tests/tensorflow    \
		--junitxml ${JUNITXML_PATH}         \
		$(DATA_ARG)

test-install-tensorflow:
	pytest tests/cross_fw/install -s --backend tf --junitxml ${JUNITXML_PATH}

test-examples-tensorflow:
	pytest tests/cross_fw/examples -s --backend tf --junitxml ${JUNITXML_PATH}

###############################################################################
# PyTorch backend
install-torch-test:
	pip install -U pip
	pip install -e .[torch] --index-url https://download.pytorch.org/whl/cu118 --extra-index-url=https://pypi.org/simple  # ticket 119128
	pip install -r tests/torch/requirements.txt --index-url https://download.pytorch.org/whl/cu118 --extra-index-url=https://pypi.org/simple
	pip install -r tests/cross_fw/install/requirements.txt
	pip install -r tests/cross_fw/examples/requirements.txt
	pip install -r examples/torch/requirements.txt --index-url https://download.pytorch.org/whl/cu118 --extra-index-url=https://pypi.org/simple

install-torch-dev: install-torch-test install-pre-commit
	pip install -r examples/post_training_quantization/torch/mobilenet_v2/requirements.txt
	pip install -r examples/post_training_quantization/torch/ssd300_vgg16/requirements.txt

install-models-hub-torch:
	pip install -U pip
	pip install -e .
	pip install -r tests/torch/models_hub_test/requirements.txt
	# Install wheel to run pip with --no-build-isolation
	pip install wheel
	pip install --no-build-isolation -r tests/torch/models_hub_test/requirements_secondary.txt


test-torch:
	pytest ${COVERAGE_ARGS} tests/torch -m "not weekly and not nightly and not models_hub" --junitxml ${JUNITXML_PATH} $(DATA_ARG)

test-torch-nightly:
	pytest ${COVERAGE_ARGS} tests/torch -m nightly --junitxml ${JUNITXML_PATH} $(DATA_ARG)

test-torch-weekly:
	pytest ${COVERAGE_ARGS} tests/torch -m weekly --junitxml ${JUNITXML_PATH} $(DATA_ARG) ${WEEKLY_MODELS_ARG}

test-install-torch-cpu:
	pytest tests/cross_fw/install -s       \
		--backend torch                     \
		--host-configuration cpu            \
		--junitxml ${JUNITXML_PATH}

test-install-torch-gpu:
	pytest tests/cross_fw/install -s        \
		--backend torch                     \
		--junitxml ${JUNITXML_PATH}

test-examples-torch:
	pytest tests/cross_fw/examples -s        \
		--backend torch                     \
		--junitxml ${JUNITXML_PATH}

test-models-hub-torch:
	pytest tests/torch/models_hub_test --junitxml ${JUNITXML_PATH}

###############################################################################
# Common part
install-common-test:
	pip install -U pip
	pip install -e .
	pip install -r tests/common/requirements.txt
	pip install -r tests/cross_fw/install/requirements.txt
	pip install -r tests/cross_fw/examples/requirements.txt

test-common:
	pytest ${COVERAGE_ARGS} tests/common $(DATA_ARG) --junitxml ${JUNITXML_PATH}

test-examples:
	pytest tests/cross_fw/examples -s --junitxml ${JUNITXML_PATH}

###############################################################################
# Pre commit check
pre-commit:
	pre-commit run -a


###############################################################################
# Fuzzing tests
install-fuzz-test: install-common-test
	pip install -r tests/cross_fw/sdl/fuzz/requirements.txt

test-fuzz:
	python tests/cross_fw/sdl/fuzz/quantize_api.py
