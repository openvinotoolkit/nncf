install-onnx-dev:
	pip install -U pip
	pip install -e .
	pip install -r nncf/experimental/onnx/requirements.txt
	pip install -r tests/onnx/requirements.txt
	pip install -r examples/experimental/onnx/requirements.txt

	# onnxruntime-openvino==1.11.0 requires numpy>=1.21.0,
	# but openvino=2022.1.0 stricts numpy<1.20.
	# Thus, we retry install numpy again for onnxruntime-openvino.
	pip install numpy==1.21.0

test-onnx:
	pytest tests/onnx
