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

import subprocess
import sys
from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import List
from typing import TypeVar
from typing import Union
from typing import Set

import numpy as np
from pathlib import Path

from tests.shared.paths import GITHUB_REPO_URL
from tests.shared.paths import PROJECT_ROOT

TensorType = TypeVar('TensorType')


def get_cli_dict_args(args):
    cli_args = {}
    for key, val in args.items():
        cli_key = '--{}'.format(str(key))
        cli_args[cli_key] = None
        if val is not None:
            cli_args[cli_key] = str(val)
    return cli_args


def create_venv_with_nncf(tmp_path: Path, package_type: str, venv_type: str, extra_reqs: Set[str] = None):
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
    extra_reqs_str = ''
    if extra_reqs is not None and extra_reqs:
        extra_reqs_str = ','.join(extra_reqs)

    if package_type == 'pip_pypi':
        subprocess.run(
            f'{pip_with_venv} install nncf[{extra_reqs_str}]', check=True, shell=True)
    elif package_type == 'pip_local':
        subprocess.run(
            f'{pip_with_venv} install {PROJECT_ROOT}[{extra_reqs_str}]', check=True, shell=True)
    elif package_type == 'pip_e_local':
        subprocess.run(
            f'{pip_with_venv} install -e {PROJECT_ROOT}[{extra_reqs_str}]', check=True, shell=True)
    elif package_type == 'pip_git_develop':
        subprocess.run(
            f'{pip_with_venv} install git+{GITHUB_REPO_URL}@develop#egg=nncf[{extra_reqs_str}]', check=True, shell=True)
    else:
        options_str = None
        if extra_reqs is not None and extra_reqs:
            options_str = ''
            for extra in extra_reqs:
                options_str += f'--{extra} '
        subprocess.run(
            '{python} {nncf_repo_root}/setup.py {package_type} {options}'.format(
                python=python_executable_with_venv,
                nncf_repo_root=PROJECT_ROOT,
                package_type=package_type,
                options=options_str if options_str is not None else ''),
            check=True,
            shell=True,
            cwd=PROJECT_ROOT)

    return venv_path



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


def telemetry_send_event_test_driver(mocker, use_nncf_fn: Callable):
    from nncf.telemetry_wrapper.telemetry import NNCFTelemetry
    telemetry_send_event_spy = mocker.spy(NNCFTelemetry, "send_event")
    use_nncf_fn()
    telemetry_send_event_spy.assert_called()
