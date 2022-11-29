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

test-install-onnx
    pytest tests/cross_fw/install/test_install.py --backend onnx
