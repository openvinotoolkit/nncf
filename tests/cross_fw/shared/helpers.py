# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import subprocess
import sys
import logging
from pathlib import Path
from typing import Callable

from tests.cross_fw.shared.paths import GITHUB_REPO_URL
from tests.cross_fw.shared.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

def is_windows() -> bool:
    return "win32" in sys.platform


def is_linux() -> bool:
    return "linux" in sys.platform


def get_cli_dict_args(args):
    cli_args = {}
    for key, val in args.items():
        cli_key = f"--{str(key)}"
        cli_args[cli_key] = None
        if val is not None:
            cli_args[cli_key] = str(val)
    return cli_args


MAP_BACKEND_PACKAGES = {
    "torch": ["torch", "torchvision"],
    "torchfx": ["torch", "torchvision"],
    "openvino": ["openvino"],
    "onnx": ["onnx", "onnxruntime"],
    "tf": ["tensorflow"],
}


def find_file_by_extension(directory: Path, extension: str) -> str:
    for file_path in directory.iterdir():
        file_path_str = str(file_path)
        if file_path_str.endswith(extension):
            return file_path_str
    msg = "NNCF package not found"
    raise FileNotFoundError(msg)


def create_venv_with_nncf(tmp_path: Path, package_type: str, venv_type: str, backends: set[str] = None):
    venv_path = tmp_path / "venv"
    venv_path.mkdir(exist_ok=True)

    python_executable_with_venv = get_python_executable_with_venv(venv_path)
    pip_with_venv = get_pip_executable_with_venv(venv_path)

    version_string = f"{sys.version_info[0]}.{sys.version_info[1]}"

    if venv_type == "virtualenv":
        virtualenv = Path(sys.executable).parent / "virtualenv"
        cmd_create_venv = f"{virtualenv} -ppython{version_string} {venv_path}"
        logger.info(f"Creating virtualenv: {cmd_create_venv}")
        subprocess.check_call(cmd_create_venv, shell=True)
    elif venv_type == "venv":
        cmd_create_venv = f"{sys.executable} -m venv {venv_path}"
        logger.info(f"Creating venv: {cmd_create_venv}")
        subprocess.check_call(cmd_create_venv, shell=True)

    cmd_upgrade_pip = f"{pip_with_venv} install --upgrade pip"
    logger.info(f"Upgrading pip: {cmd_upgrade_pip}")
    subprocess.check_call(cmd_upgrade_pip, shell=True)

    cmd_upgrade_tools = f"{pip_with_venv} install --upgrade wheel setuptools"
    logger.info(f"Upgrading wheel and setuptools: {cmd_upgrade_tools}")
    subprocess.check_call(cmd_upgrade_tools, shell=True)

    if package_type in ["build_s", "build_w"]:
        cmd_install_build = f"{pip_with_venv} install build"
        logger.info(f"Installing build: {cmd_install_build}")
        subprocess.check_call(cmd_install_build, shell=True)

    run_path = tmp_path / "run"
    run_path.mkdir(exist_ok=True)

    if package_type in ["build_s", "build_w"]:
        dist_path = tmp_path / "dist"
        dist_path.mkdir(exist_ok=True)
        build_path = tmp_path / "build"
        build_path.mkdir(exist_ok=True)

    if package_type == "pip_pypi":
        run_cmd_line = f"{pip_with_venv} install nncf"
    elif package_type == "pip_local":
        run_cmd_line = f"{pip_with_venv} install {PROJECT_ROOT}"
    elif package_type == "pip_e_local":
        run_cmd_line = f"{pip_with_venv} install -e {PROJECT_ROOT}"
    elif package_type == "pip_git_develop":
        run_cmd_line = f"{pip_with_venv} install git+{GITHUB_REPO_URL}@develop#egg=nncf"
    elif package_type == "build_s":
        run_cmd_line = f"{python_executable_with_venv} -m build -s --outdir {dist_path}"
    elif package_type == "build_w":
        run_cmd_line = f"{python_executable_with_venv} -m build -w --outdir {dist_path}"
    else:
        msg = f"Invalid package type: {package_type}"
        raise ValueError(msg)

    logger.info(f"Running package command: {run_cmd_line}")
    subprocess.run(run_cmd_line, check=True, shell=True, cwd=PROJECT_ROOT)

    if package_type in ["build_s", "build_w"]:
        package_path = find_file_by_extension(dist_path, ".tar.gz" if package_type == "build_s" else ".whl")
        cmd_install_package = f"{pip_with_venv} install {package_path}"
        logger.info(f"Installing package: {cmd_install_package}")
        subprocess.run(cmd_install_package, check=True, shell=True)

    if backends:
        # Install backend specific packages with according version from constraints.txt
        packages = [item for b in backends for item in MAP_BACKEND_PACKAGES[b]]
        extra_reqs = " ".join(packages)
        cmd_install_backends = f"{pip_with_venv} install {extra_reqs} -c {PROJECT_ROOT}/constraints.txt"
        logger.info(f"Installing backend packages: {cmd_install_backends}")
        subprocess.run(
            cmd_install_backends,
            check=True,
            shell=True,
            cwd=PROJECT_ROOT,
        )

    return venv_path


def telemetry_send_event_test_driver(mocker, use_nncf_fn: Callable):
    from nncf.telemetry import telemetry

    telemetry_send_event_spy = mocker.spy(telemetry, "send_event")
    use_nncf_fn()
    telemetry_send_event_spy.assert_called()


def get_python_executable_with_venv(venv_path: Path) -> str:
    if is_linux():
        python_executable_with_venv = f". {venv_path}/bin/activate && {venv_path}/bin/python"
    elif is_windows():
        python_executable_with_venv = f" {venv_path}\\Scripts\\activate && python"

    return python_executable_with_venv


def get_pip_executable_with_venv(venv_path: Path) -> str:
    if is_linux():
        pip_with_venv = f". {venv_path}/bin/activate && {venv_path}/bin/pip"
    elif is_windows():
        pip_with_venv = f" {venv_path}\\Scripts\\activate && python -m pip"
    return pip_with_venv


def remove_line_breaks(s: str) -> str:
    return s.replace("\r\n", "").replace("\n", "")
