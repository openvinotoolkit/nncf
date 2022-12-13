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

test-e2e-ptq-ov-ep-only:
	pytest tests/onnx -m e2e_ptq --model-dir /models --data-dir /datasets --output-dir outputs --anno-dir /annots \
    --junitxml outputs/nncf-tests.xml --ptq-size 100 --eval-size 1000

test-e2e-ptq-ep-only:
	pytest tests/onnx -m e2e_ptq --model-dir /models --data-dir /datasets --output-dir outputs --anno-dir /annots \
    --junitxml outputs/nncf-tests.xml --ptq-size 100 --eval-size 1000 --enable-cpu-ep --disable-ov-ep

test-e2e-ptq-ov-cpu-eps:
	pytest tests/onnx -m e2e_ptq --model-dir /models --data-dir /datasets --output-dir outputs --anno-dir /annots  \
	--junitxml outputs/nncf-tests.xml --ptq-size 100 --eval-size 1000 --enable-cpu-ep