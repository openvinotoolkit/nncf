install-onnx-dev:
	pip install -U pip
	pip install -e .[onnx]
	pip install -r tests/onnx/requirements.txt
	pip install -r tests/onnx/benchmarking/requirements.txt
	pip install -r examples/post_training_quantization/onnx/mobilenet_v2/requirements.txt

	# Install pylint
	pip install pylint==2.13.9

test-onnx:
	pytest tests/onnx --junitxml nncf-tests.xml

PYFILES := $(shell find examples/post_training_quantization/onnx -type f -name "*.py")
pylint-onnx:
	pylint --rcfile .pylintrc               \
		nncf/experimental/onnx              \
		nncf/quantization                   \
		tests/onnx                          \
		$(PYFILES)                         

test-install-onnx:
	pytest tests/cross_fw/install/ --backend onnx

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
	export LD_LIBRARY_PATH=LD_LIBRARY_PATH:${NNCF_DIR}bin/intel64/Release
	pip install ${NNCF_DIR}/tests/onnx/openvino-2022.3.0-8784-cp38-cp38-manylinux_2_31_x86_64.whl
	pip install ${NNCF_DIR}/tests/onnx/openvino_dev-2022.3.0-8784-py3-none-any.whl
	pip install ${NNCF_DIR}/tests/onnx/onnxruntime_openvino-1.14.0-cp38-cp38-linux_x86_64.whl
	pip install numpy==1.23.1
