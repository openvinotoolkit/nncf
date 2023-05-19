import os
from pathlib import Path

import pytest
import torch

from nncf.torch.binarization.extensions import BinarizedFunctionsCPU
from nncf.torch.binarization.extensions import BinarizedFunctionsCUDA
from nncf.torch.extensions import EXTENSION_LOAD_TIMEOUT_ENV_VAR
from nncf.torch.extensions import ExtensionLoaderTimeoutException
from tests.shared.isolation_runner import ISOLATION_RUN_ENV_VAR
from tests.shared.isolation_runner import run_pytest_case_function_in_separate_process


@pytest.mark.skipif(ISOLATION_RUN_ENV_VAR not in os.environ, reason="Should be run via isolation proxy")
def test_timeout_extension_loader_isolated(tmp_path, use_cuda):
    if not torch.cuda.is_available() and use_cuda is True:
        pytest.skip("Skipping CUDA test cases for CPU only setups")

    quant_func = BinarizedFunctionsCPU if use_cuda else BinarizedFunctionsCUDA

    os.environ[EXTENSION_LOAD_TIMEOUT_ENV_VAR] = "1"
    os.environ["TORCH_EXTENSIONS_DIR"] = tmp_path.as_posix()

    # pylint: disable=protected-access
    build_dir = Path(quant_func._loader.get_build_dir())
    lock_file = build_dir / "lock"
    lock_file.touch()
    with pytest.raises(ExtensionLoaderTimeoutException):
        quant_func.get("ActivationBinarize_forward")


def test_timeout_extension_loader():
    run_pytest_case_function_in_separate_process(test_timeout_extension_loader_isolated)
