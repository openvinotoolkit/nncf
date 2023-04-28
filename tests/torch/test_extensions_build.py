import os
import pathlib
import shutil
import subprocess

import pytest
import torch

from tests.shared.command import Command
from tests.shared.helpers import create_venv_with_nncf
from tests.shared.helpers import get_python_executable_with_venv
from tests.shared.paths import TEST_ROOT

EXTENSIONS_BUILD_FILENAME = "extensions_build_checks.py"


def test_force_cuda_build(tmp_path):
    """
    Check that CUDA Extensions weren't initially built and \
    then with TORCH_CUDA_ARCH_LIST were forced to be built
    """
    venv_path = create_venv_with_nncf(tmp_path, package_type="pip_local", venv_type="venv", extra_reqs={"torch"})
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        try:
            nvcc = subprocess.check_output(["which", "nvcc"])
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except subprocess.CalledProcessError:
            if not cuda_home:
                cuda_home = "/usr/local/cuda"
                if not os.path.exists(cuda_home):
                    cuda_home = None
        if not cuda_home and not torch.cuda.is_available():
            pytest.skip("There is no CUDA on the machine. The test will be skipped")

    torch_build_dir = tmp_path / "extensions"

    env_variables = {"CUDA_VISIBLE_DEVICES": "", "TORCH_EXTENSIONS_DIR": str(torch_build_dir)}

    python_executable_with_venv = get_python_executable_with_venv(venv_path)

    run_path = tmp_path / "run"

    shutil.copy(TEST_ROOT / "torch" / EXTENSIONS_BUILD_FILENAME, run_path)

    torch_ext_dir = pathlib.Path(torch_build_dir)
    assert not torch_ext_dir.exists()

    mode = "cpu"

    command = Command(
        "{} {}/extensions_build_checks.py {}".format(python_executable_with_venv, run_path, mode),
        cwd=run_path,
        env=env_variables,
    )
    command.run()

    version_command = Command(
        '{} -c "import torch; print(torch.__version__)"'.format(python_executable_with_venv),
        cwd=run_path,
        env=env_variables,
    )
    version_command.run()
    torch_version = version_command.output[0].replace("\n", "")

    cpu_ext_dir = torch_ext_dir / "nncf" / "quantized_functions_cpu" / torch_version
    assert cpu_ext_dir.exists()
    cpu_ext_so = cpu_ext_dir / "quantized_functions_cpu.so"
    assert cpu_ext_so.exists()

    cuda_ext_dir = torch_ext_dir / "nncf" / "quantized_functions_cuda" / torch_version
    assert not cuda_ext_dir.exists()
    cuda_ext_so = cuda_ext_dir / "quantized_functions_cuda.so"
    assert not cuda_ext_so.exists()

    cpu_ext_dir = torch_ext_dir / "nncf" / "binarized_functions_cpu" / torch_version
    assert cpu_ext_dir.exists()
    cpu_ext_so = cpu_ext_dir / "binarized_functions_cpu.so"
    assert cpu_ext_so.exists()

    cuda_ext_dir = torch_ext_dir / "nncf" / "binarized_functions_cuda" / torch_version
    assert not cuda_ext_dir.exists()
    cuda_ext_so = cuda_ext_dir / "nncf" / torch_version / "binarized_functions_cuda.so"
    assert not cuda_ext_so.exists()

    mode = "cuda"

    command = Command(
        "{} {}/extensions_build_checks.py {}".format(python_executable_with_venv, run_path, mode),
        cwd=run_path,
        env=env_variables,
    )
    command.run()

    cuda_ext_dir = torch_ext_dir / "nncf" / "quantized_functions_cuda" / torch_version
    assert cuda_ext_dir.exists()
    cuda_ext_so = cuda_ext_dir / "quantized_functions_cuda.so"
    assert cuda_ext_so.exists()

    cuda_ext_dir = torch_ext_dir / "nncf" / "binarized_functions_cuda" / torch_version
    assert cuda_ext_dir.exists()
    cuda_ext_so = cuda_ext_dir / "binarized_functions_cuda.so"
    assert cuda_ext_so.exists()
