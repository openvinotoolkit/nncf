import inspect
import os
import subprocess
from typing import Callable
from typing import Tuple

import pytest

ISOLATION_RUN_ENV_VAR = "ISOLATION_RUN"
def run_pytest_case_function_in_separate_process(fn: Callable) -> Tuple[int, str, str]:
    """
    Use this function for launching test cases that rely on global object behaviour such as module import
    order that cannot be reliably reset and/or isolated using regular pytest functionality. This will
    launch a separate pytest process that will run only the test function passed as the parameter. If the
    pytest run failed, this function will trigger pytest.fail().
    :param fn - The function object corresponding to a pytest test function defined elsewhere.
    :returns A tuple of return code for the pytest invocation, and string representations of the stdout and stderr
    pipe outputs of the pytest invoсation.
    """
    filename = inspect.getfile(fn)
    func_name = fn.__name__
    env = os.environ.copy()
    env[ISOLATION_RUN_ENV_VAR] = "1"
    with subprocess.Popen(f'pytest -s {filename} -k {func_name}',
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, shell=True,
                          bufsize=1,
                          env=env) as p:
        stdout, stderr = p.communicate()

    stdout = stdout.decode("utf-8") if stdout is not None else ''
    stderr = stderr.decode("utf-8") if stderr is not None else ''
    if p.returncode != 0:
        pytest.fail(f"pytest invocation failed with exit code {p.returncode}\n"
                    f"STDOUT:\n"
                    f"{stdout}\n"
                    f"STDERR:\n"
                    f"{stderr}\n")
    return p.returncode, stdout, stderr
