install-onnx-dev:
	pip install -U pip
	pip install -e .[onnx]
	pip install -r tests/onnx/requirements.txt
	pip install -r tests/onnx/benchmarking/requirements.txt

	# Install pylint
	pip install pylint==2.13.9

test-onnx:
	pytest tests/onnx --junitxml nncf-tests.xml

pylint-onnx:
	pylint --rcfile .pylintrc				\
		nncf/experimental/onnx				\
		nncf/quantization					\
		tests/onnx examples/experimental/onnx

test-install-onnx:
	pytest tests/cross_fw/install/ --backend onnx

ONNX_PTQ_SIZE=100
ONNX_PTQ_EVAL_SIZE=1000
ONNX_E2E_OPTIONS=--model-dir /models --data-dir /datasets --output-dir outputs --anno-dir /annots \
    --junitxml outputs/nncf-tests.xml --ptq-size $(ONNX_PTQ_SIZE) --eval-size $(ONNX_PTQ_EVAL_SIZE)

test-e2e-ptq-ov-ep-only:
	pytest tests/onnx -m e2e_ptq $(ONNX_E2E_OPTIONS)

test-e2e-ptq-cpu-ep-only:
	pytest tests/onnx -m e2e_ptq $(ONNX_E2E_OPTIONS) --enable-cpu-ep --disable-ov-ep

test-e2e-ptq-ov-cpu-eps:
	pytest tests/onnx -m e2e_ptq $(ONNX_E2E_OPTIONS) --enable-cpu-ep