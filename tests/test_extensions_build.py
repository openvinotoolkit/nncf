import os
import subprocess
import sys
import shutil
import pathlib

from tests.conftest import TEST_ROOT, PROJECT_ROOT

EXTENSIONS_BUILD_FILENAME = 'extensions_build_checks.py'


def test_force_cuda_build(tmp_path):
    venv_path = tmp_path / 'venv'
    venv_path.mkdir()

    ext_path = tmp_path / 'extensions'
    export_env_variables = "export CUDA_VISIBLE_DEVICES='' export TORCH_EXTENSIONS_DIR={}".format(ext_path)

    python_executable_with_venv = ". {0}/bin/activate && {1} && {0}/bin/python".format(venv_path, export_env_variables)

    version_string = "{}.{}".format(sys.version_info[0], sys.version_info[1])

    subprocess.call("virtualenv -ppython{} {}".format(version_string, venv_path), shell=True)

    subprocess.run(
        "{python} {nncf_repo_root}/setup.py develop".format(
            python=python_executable_with_venv,
            nncf_repo_root=PROJECT_ROOT),
        check=True,
        shell=True,
        cwd=PROJECT_ROOT)

    run_path = tmp_path / 'run'
    run_path.mkdir()

    shutil.copy(TEST_ROOT / EXTENSIONS_BUILD_FILENAME, run_path)

    subprocess.run(
        "{} {}/extensions_build_checks.py {}".format(python_executable_with_venv, run_path, ext_path),
        check=True, shell=True, cwd=run_path)


def test_if_cuda_was_built_without_gpu(torch_build_dir):
    '''
    Check that CUDA Extensions weren't initially built and \
    then with TORCH_CUDA_ARCH_LIST were forced to be built
    '''
    torch_ext_dir = pathlib.Path(torch_build_dir)
    assert not torch_ext_dir.exists()

    # Do not remove - the import here is for testing purposes.
    # pylint: disable=wrong-import-position
    from nncf import force_build_cuda_extensions
    # pylint: enable=wrong-import-position

    cpu_ext_dir = (torch_ext_dir / 'quantized_functions_cpu')
    assert cpu_ext_dir.exists()
    cpu_ext_so = (cpu_ext_dir / 'quantized_functions_cpu.so')
    assert cpu_ext_so.exists()

    cuda_ext_dir = (torch_ext_dir / 'quantized_functions_cuda')
    assert not cuda_ext_dir.exists()
    cuda_ext_so = (cuda_ext_dir / 'quantized_functions_cuda.so')
    assert not cuda_ext_so.exists()

    # Set CUDA Architecture
    # See cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
    os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5+PTX'
    force_build_cuda_extensions()

    cuda_ext_dir = (torch_ext_dir / 'quantized_functions_cuda')
    assert cuda_ext_dir.exists()
    cuda_ext_so = (cuda_ext_dir / 'quantized_functions_cuda.so')
    assert cuda_ext_so.exists()
