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
	git clone https://github.com/openvinotoolkit/openvino.git
	cd openvino && git checkout 74758c0ca5fc7b76e21b490406cb2d81c048dc4e
	cd openvino && git submodule update --init --recursive && chmod +x install_build_dependencies.sh
	cd openvino && sudo -E bash ./install_build_dependencies.sh
	cd openvino && mkdir build
	cd openvino && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON -DENABLE_WHEEL=ON -DENABLE_OV_ONNX_FRONTEND=ON ..
	cd openvino && cd build && make --jobs=$(shell nproc --all)	
	pip install ./openvino/build/wheels/openvino-2022.3.0-8784-cp38-cp38-manylinux_2_31_x86_64.whl
	pip install ./openvino/build/wheels/openvino_dev-2022.3.0-8784-py3-none-any.whl
	export LD_LIBRARY_PATH=LD_LIBRARY_PATH:./openvino/bin/intel64/Release
	git lfs pull --include ${NNCF_DIR}tests/onnx/onnxruntime_openvino-1.14.0-cp38-cp38-linux_x86_64.whl
	pip install ${NNCF_DIR}/tests/onnx/onnxruntime_openvino-1.14.0-cp38-cp38-linux_x86_64.whl
	pip install numpy==1.23.1
