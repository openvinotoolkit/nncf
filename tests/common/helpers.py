"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import shutil
import subprocess
import sys
from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import List
from typing import TypeVar
from typing import Union

import numpy as np

from tests.common.paths import GITHUB_REPO_URL
from tests.common.paths import PROJECT_ROOT
from tests.common.paths import TEST_ROOT

TensorType = TypeVar('TensorType')


def get_cli_dict_args(args):
    cli_args = {}
    for key, val in args.items():
        cli_key = '--{}'.format(str(key))
        cli_args[cli_key] = None
        if val is not None:
            cli_args[cli_key] = str(val)
    return cli_args


def create_venv_with_nncf(tmp_path, package_type, venv_type, extra_reqs):
    venv_path = tmp_path / 'venv'
    venv_path.mkdir()

    python_executable_with_venv = '. {0}/bin/activate && {0}/bin/python'.format(venv_path)
    pip_with_venv = '. {0}/bin/activate && {0}/bin/pip'.format(venv_path)

    version_string = '{}.{}'.format(sys.version_info[0], sys.version_info[1])
    if venv_type == 'virtualenv':
        subprocess.call('virtualenv -ppython{} {}'.format(version_string, venv_path), shell=True)
    elif venv_type == 'venv':
        subprocess.call('python{} -m venv {}'.format(version_string, venv_path), shell=True)
        subprocess.call('{} install --upgrade pip'.format(pip_with_venv), shell=True)
        subprocess.call('{} install wheel'.format(pip_with_venv), shell=True)

    run_path = tmp_path / 'run'
    run_path.mkdir()

    if package_type == 'pip_pypi':
        subprocess.run(
            f'{pip_with_venv} install nncf[{extra_reqs}]', check=True, shell=True)
    elif package_type == 'pip_local':
        subprocess.run(
            f'{pip_with_venv} install {PROJECT_ROOT}[{extra_reqs}]', check=True, shell=True)
    elif package_type == 'pip_e_local':
        subprocess.run(
            f'{pip_with_venv} install -e {PROJECT_ROOT}[{extra_reqs}]', check=True, shell=True)
    elif package_type == 'pip_git_develop':
        subprocess.run(
            f'{pip_with_venv} install git+{GITHUB_REPO_URL}@develop#egg=nncf[{extra_reqs}]', check=True, shell=True)
    else:
        subprocess.run(
            '{python} {nncf_repo_root}/setup.py {package_type} {options}'.format(
                python=python_executable_with_venv,
                nncf_repo_root=PROJECT_ROOT,
                package_type=package_type,
                options=f'--{extra_reqs}' if extra_reqs else ''),
            check=True,
            shell=True,
            cwd=PROJECT_ROOT)

    return venv_path


def run_install_checks(venv_path, tmp_path, package_type, test_dir, install_type=''):
    python_executable_with_venv = '. {0}/bin/activate && {0}/bin/python'.format(venv_path)
    pip_with_venv = '. {0}/bin/activate && {0}/bin/pip'.format(venv_path)

    run_path = tmp_path / 'run'

    shutil.copy(TEST_ROOT / test_dir / 'install_checks.py', run_path)

    # Do additional install step for sdist/bdist packages
    if package_type == 'sdist':
        for file_name in os.listdir(os.path.join(PROJECT_ROOT, 'dist')):
            if file_name.endswith('.tar.gz'):
                package_name = file_name
                break
        else:
            raise FileNotFoundError('NNCF package not found')

        option = 'tf' if test_dir == 'tensorflow' else 'torch'
        subprocess.run(
            '{} install {}/dist/{}[{}] '.format(pip_with_venv,
                                                PROJECT_ROOT,
                                                package_name,
                                                option),
            check=True, shell=True)
    elif package_type == "bdist_wheel":
        subprocess.run(
            "{} install {}/dist/*.whl ".format(pip_with_venv, PROJECT_ROOT), check=True, shell=True)

    if install_type.lower() == 'cpu':
        install_mode = 'cpu'
    elif install_type.lower() == 'gpu':
        install_mode = 'cuda'
    else:
        install_mode = ''
    subprocess.run(
        '{} {}/install_checks.py {} {}'.format(python_executable_with_venv,
                                               run_path,
                                               install_mode,
                                               package_type),
        check=True, shell=True, cwd=run_path)


class BaseTensorListComparator(ABC):
    @classmethod
    @abstractmethod
    def _to_numpy(cls, tensor: TensorType) -> np.ndarray:
        pass

    @classmethod
    def _check_assertion(cls, test: Union[TensorType, List[TensorType]],
                         reference: Union[TensorType, List[TensorType]],
                         assert_fn: Callable[[np.ndarray, np.ndarray], bool]):
        if not isinstance(test, list):
            test = [test]
        if not isinstance(reference, list):
            reference = [reference]
        assert len(test) == len(reference)

        for x, y in zip(test, reference):
            x = cls._to_numpy(x)
            y = cls._to_numpy(y)
            assert_fn(x, y)

    @classmethod
    def check_equal(cls, test: Union[TensorType, List[TensorType]], reference: Union[TensorType, List[TensorType]],
                    rtol: float = 1e-1, atol=0):
        cls._check_assertion(test, reference, lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol, atol=atol))

    @classmethod
    def check_not_equal(cls, test: Union[TensorType, List[TensorType]], reference: Union[TensorType, List[TensorType]],
                        rtol: float = 1e-4):
        cls._check_assertion(test, reference, lambda x, y: np.testing.assert_raises(AssertionError,
                                                                                    np.testing.assert_allclose,
                                                                                    x, y, rtol=rtol))

    @classmethod
    def check_less(cls, test: Union[TensorType, List[TensorType]], reference: Union[TensorType, List[TensorType]],
                   rtol=1e-4):
        cls.check_not_equal(test, reference, rtol=rtol)
        cls._check_assertion(test, reference, np.testing.assert_array_less)

    @classmethod
    def check_greater(cls, test: Union[TensorType, List[TensorType]], reference: Union[TensorType, List[TensorType]],
                      rtol=1e-4):
        cls.check_not_equal(test, reference, rtol=rtol)
        cls._check_assertion(test, reference, lambda x, y: np.testing.assert_raises(AssertionError,
                                                                                    np.testing.assert_array_less,
                                                                                    x, y))
