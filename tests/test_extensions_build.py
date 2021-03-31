import os
import subprocess
import pytest
import pathlib
import shutil

import torch

from tests.conftest import TEST_ROOT
from tests.test_sanity_sample import Command

EXTENSIONS_BUILD_FILENAME = 'extensions_build_checks.py'


@pytest.mark.parametrize("venv_type, package_type,install_type",
                         [('venv', 'develop', 'GPU')])
def test_force_cuda_build(tmp_venv_with_nncf, install_type, tmp_path, package_type):
    '''
    Check that CUDA Extensions weren't initially built and \
    then with TORCH_CUDA_ARCH_LIST were forced to be built
    '''
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        try:
            nvcc = subprocess.check_output(['which', 'nvcc'])
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except subprocess.CalledProcessError:
            if not cuda_home:
                cuda_home = '/usr/local/cuda'
                if not os.path.exists(cuda_home):
                    cuda_home = None
        if not cuda_home and not torch.cuda.is_available():
            pytest.skip('There is no CUDA on the machine. The test will be skipped')

    venv_path = tmp_venv_with_nncf

    torch_build_dir = tmp_path / 'extensions'
    export_env_variables = "export CUDA_VISIBLE_DEVICES='' export TORCH_EXTENSIONS_DIR={}".format(torch_build_dir)

    python_executable_with_venv = ". {0}/bin/activate && {1} && {0}/bin/python".format(venv_path, export_env_variables)

    run_path = tmp_path / 'run'

    shutil.copy(TEST_ROOT / EXTENSIONS_BUILD_FILENAME, run_path)

    torch_ext_dir = pathlib.Path(torch_build_dir)
    assert not torch_ext_dir.exists()

    mode = 'cpu'

    command = Command("{} {}/extensions_build_checks.py {}".format(python_executable_with_venv, run_path, mode),
                      path=run_path)
    command.run()

    cpu_ext_dir = (torch_ext_dir / 'quantized_functions_cpu')
    assert cpu_ext_dir.exists()
    cpu_ext_so = (cpu_ext_dir / 'quantized_functions_cpu.so')
    assert cpu_ext_so.exists()

    cuda_ext_dir = (torch_ext_dir / 'quantized_functions_cuda')
    assert not cuda_ext_dir.exists()
    cuda_ext_so = (cuda_ext_dir / 'quantized_functions_cuda.so')
    assert not cuda_ext_so.exists()

    cpu_ext_dir = (torch_ext_dir / 'binarized_functions_cpu')
    assert cpu_ext_dir.exists()
    cpu_ext_so = (cpu_ext_dir / 'binarized_functions_cpu.so')
    assert cpu_ext_so.exists()

    cuda_ext_dir = (torch_ext_dir / 'binarized_functions_cuda')
    assert not cuda_ext_dir.exists()
    cuda_ext_so = (cuda_ext_dir / 'binarized_functions_cuda.so')
    assert not cuda_ext_so.exists()

    mode = 'cuda'

    command = Command("{} {}/extensions_build_checks.py {}".format(python_executable_with_venv, run_path, mode),
                      path=run_path)
    command.run()

    cuda_ext_dir = (torch_ext_dir / 'quantized_functions_cuda')
    assert cuda_ext_dir.exists()
    cuda_ext_so = (cuda_ext_dir / 'quantized_functions_cuda.so')
    assert cuda_ext_so.exists()

    cuda_ext_dir = (torch_ext_dir / 'binarized_functions_cuda')
    assert cuda_ext_dir.exists()
    cuda_ext_so = (cuda_ext_dir / 'binarized_functions_cuda.so')
    assert cuda_ext_so.exists()
