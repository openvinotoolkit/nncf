# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps
from importlib import import_module
from typing import Any, Callable, ParamSpec, TypeVar

from nncf.common.logging import nncf_logger

IMPORTED_DEPENDENCIES: dict[str, bool] = {}

P = ParamSpec("P")
T = TypeVar("T")

_PLOTS_EXTRA_HINT = (
    "Please install NNCF package with plots extra. Use one of the following commands "
    '"pip install .[plots]" running from the repository root directory or "pip install nncf[plots]"'
)


def skip_if_dependency_unavailable(dependencies: list[str]) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """
    Decorator factory to skip a noreturn function if dependencies are not met.

    :param dependencies: A list of dependencies
    :return: A decorator
    """

    def wrap(func: Callable[..., None]) -> Callable[..., None]:
        def wrapped_f(*args: Any, **kwargs: Any):  # type: ignore
            for libname in dependencies:
                if libname in IMPORTED_DEPENDENCIES:
                    if IMPORTED_DEPENDENCIES[libname]:
                        continue
                    break
                try:
                    _ = import_module(libname)
                    IMPORTED_DEPENDENCIES[libname] = True
                except ImportError as ex:
                    nncf_logger.warning(f"{ex.msg} {_PLOTS_EXTRA_HINT}")
                    IMPORTED_DEPENDENCIES[libname] = False
                    break
            else:
                return func(*args, **kwargs)
            return None

        return wrapped_f

    return wrap


def raise_if_dependency_unavailable(dependencies: list[str]) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator factory to raise an informative error if dependencies are not met.

    :param dependencies: A list of module names required by the decorated function.
    :return: A decorator that raises ImportError with an installation hint when a dependency is missing.
    """

    def wrap(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapped_f(*args: P.args, **kwargs: P.kwargs) -> T:
            for libname in dependencies:
                if IMPORTED_DEPENDENCIES.get(libname, False):
                    continue
                try:
                    _ = import_module(libname)
                    IMPORTED_DEPENDENCIES[libname] = True
                except ImportError as ex:
                    IMPORTED_DEPENDENCIES[libname] = False
                    msg = f"{ex.msg} {_PLOTS_EXTRA_HINT}"
                    raise ImportError(msg) from ex
            return func(*args, **kwargs)

        return wrapped_f

    return wrap
