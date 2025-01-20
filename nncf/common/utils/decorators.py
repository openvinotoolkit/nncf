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

from importlib import import_module
from typing import Any, Callable, Dict, List

from nncf.common.logging import nncf_logger

IMPORTED_DEPENDENCIES: Dict[str, bool] = {}


def skip_if_dependency_unavailable(dependencies: List[str]) -> Callable[[Callable[..., None]], Callable[..., None]]:
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
                    nncf_logger.warning(
                        f"{ex.msg} Please install NNCF package with plots "
                        "extra. Use one of the following commands "
                        '"pip install .[plots]" running from the repository '
                        'root directory or "pip install nncf[plots]"'
                    )
                    IMPORTED_DEPENDENCIES[libname] = False
                    break
            else:
                return func(*args, **kwargs)
            return None

        return wrapped_f

    return wrap
