ARGS := $(wordlist 2, $(words $(MAKECMDGOALS)), $(MAKECMDGOALS))

ifdef NUM_WORKERS
	NUM_WORKERS_ARG := -n${NUM_WORKERS}
endif

GENERAL_PYTEST_ARGS := $(NUM_WORKERS_ARG) -ra --durations=30
CMD_INSTALL := uv pip install
CMD_PYTEST := python -m pytest


###############################################################################
# Common part
###############################################################################
install-common:
	$(CMD_INSTALL) -e . -r tests/common/requirements.txt

test-common:
	$(CMD_PYTEST) tests/common $(GENERAL_PYTEST_ARGS) $(ARGS)


###############################################################################
# OpenVINO
###############################################################################
install-openvino:
	$(CMD_INSTALL) -e . -r tests/openvino/requirements.txt

test-openvino:
	ONEDNN_MAX_CPU_ISA=AVX2 $(CMD_PYTEST) tests/openvino $(GENERAL_PYTEST_ARGS) $(ARGS)


###############################################################################
# PyTorch
###############################################################################
install-torch:
	$(CMD_INSTALL) -e . -r tests/torch/requirements.txt

test-torch:
	$(CMD_PYTEST) tests/torch $(GENERAL_PYTEST_ARGS) $(ARGS)

test-torch-cpu:
	$(CMD_PYTEST) tests/torch -m "not cuda and not long" $(GENERAL_PYTEST_ARGS) $(ARGS)

test-torch-gpu:
	$(CMD_PYTEST) tests/torch -m "cuda and not long" $(GENERAL_PYTEST_ARGS) $(ARGS)

test-torch-long:
	$(CMD_PYTEST) tests/torch -m "long" $(GENERAL_PYTEST_ARGS) $(ARGS)


###############################################################################
# ONNX
###############################################################################
install-onnx:
	$(CMD_INSTALL) -e . -r tests/onnx/requirements.txt

test-onnx:
	$(CMD_PYTEST) tests/onnx $(GENERAL_PYTEST_ARGS) $(ARGS)


###############################################################################
# Examples
###############################################################################
install-examples:
	$(CMD_INSTALL) -r tests/cross_fw/examples/requirements.txt

test-examples:
	$(CMD_PYTEST) $(GENERAL_PYTEST_ARGS) -s tests/cross_fw/examples $(ARGS)


###############################################################################
# Conformance
###############################################################################
install-conformance:
	$(CMD_INSTALL) -e . -r tests/post_training/requirements.txt

test-conformance-ptq:
	$(CMD_PYTEST) $(GENERAL_PYTEST_ARGS) -s tests/post_training/test_quantize_conformance.py::test_ptq_quantization $(ARGS)

test-conformance-wc:
	$(CMD_PYTEST) $(GENERAL_PYTEST_ARGS) -s tests/post_training/test_quantize_conformance.py::test_weight_compression $(ARGS)


###############################################################################
# Docs
###############################################################################
install-docs:
	$(CMD_INSTALL) -r tests/docs/requirements.txt

test-docs:
	$(CMD_PYTEST) tests/docs $(GENERAL_PYTEST_ARGS) $(ARGS)


###############################################################################
# Tools
###############################################################################
install-tools:
	$(CMD_INSTALL) -e . -r tests/tools/requirements.txt

test-tools:
	$(CMD_PYTEST) tests/tools $(GENERAL_PYTEST_ARGS) $(ARGS)


###############################################################################
# Linters
###############################################################################
install-pre-commit:
	pip install pre-commit

pre-commit:
	pre-commit run -a
