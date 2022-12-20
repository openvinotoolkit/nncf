
JUNITXML_PATH ?= nncf-tests.xml
MODELS_DIR ?= /models
DATA_DIR ?= /datasets
OUTPUT_DIR ?= /output
ANNO_DIR ?= /annots

install-onnx-dev:
	pip install -U pip
	pip install -e .[onnx]
	pip install -r tests/onnx/requirements.txt
	pip install -r tests/onnx/benchmarking/requirements.txt
	pip install -r examples/post_training_quantization/onnx/mobilenet_v2/requirements.txt

	# Install pylint
	pip install pylint==2.13.9

test-onnx:
	pytest tests/onnx --junitxml ${JUNITXML_PATH}

PYFILES := $(shell find examples/post_training_quantization/onnx -type f -name "*.py")
pylint-onnx:
	pylint --rcfile .pylintrc               \
		nncf/experimental/onnx              \
		nncf/quantization                   \
		tests/onnx                          \
		$(PYFILES)                         

test-install-onnx:
	pytest tests/cross_fw/install/ --backend onnx --junitxml ${JUNITXML_PATH}

ONNX_E2E_PTQ_SIZE ?= 100
ONNX_E2E_PTQ_EVAL_SIZE ?= 1000
ONNX_E2E_OPTIONS=--model-dir ${MODELS_DIR} --data-dir ${DATA_DIR}  \
				 --output-dir ${OUTPUT_DIR} --anno-dir ${ANNO_DIR} \
                 --junitxml ${JUNITXML_PATH} --ptq-size ${ONNX_E2E_PTQ_SIZE} \
                 --eval-size ${ONNX_E2E_PTQ_EVAL_SIZE}

test-e2e-ptq-ov-ep-only:
	pytest tests/onnx -m e2e_ptq ${ONNX_E2E_OPTIONS}

test-e2e-ptq-cpu-ep-only:
	pytest tests/onnx -m e2e_ptq ${ONNX_E2E_OPTIONS} --enable-cpu-ep --disable-ov-ep

