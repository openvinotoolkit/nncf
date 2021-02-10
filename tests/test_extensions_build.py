import subprocess
import sys
import shutil

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
