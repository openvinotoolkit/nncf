# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import inspect
import warnings
from typing import Callable, Type, TypeVar

from packaging import version


def warning_deprecated(msg: str) -> None:
    # Note: must use FutureWarning in order not to get suppressed by default
    warnings.warn(msg, FutureWarning, stacklevel=2)


ClassOrFn = TypeVar("ClassOrFn", Callable, Type)


class deprecated:
    """
    A decorator for marking function calls or class instantiations as deprecated. A call to the marked function or an
    instantiation of an object of the marked class will trigger a `FutureWarning`. If a class is marked as
    @deprecated, only the instantiations will trigger a warning, but static attribute accesses or method calls will not.
    """

    def __init__(self, msg: str = None, start_version: str = None, end_version: str = None):
        """
        :param msg: Custom message to be added after the boilerplate deprecation text.
        """
        self.msg = msg
        self.start_version = version.parse(start_version) if start_version is not None else None
        self.end_version = version.parse(end_version) if end_version is not None else None

    def __call__(self, fn_or_class: ClassOrFn) -> ClassOrFn:
        name = fn_or_class.__module__ + "." + fn_or_class.__name__
        if inspect.isclass(fn_or_class):
            fn_or_class.__init__ = self._get_wrapper(fn_or_class.__init__, name)
            return fn_or_class
        return self._get_wrapper(fn_or_class, name)

    def _get_wrapper(self, fn_to_wrap: Callable, name: str) -> Callable:
        @functools.wraps(fn_to_wrap)
        def wrapped(*args, **kwargs):
            msg = f"Usage of {name} is deprecated "
            if self.start_version is not None:
                msg += f"starting from NNCF v{str(self.start_version)} "
            msg += "and will be removed in "
            if self.end_version is not None:
                msg += f"NNCF v{str(self.end_version)}."
            else:
                msg += "a future NNCF version."
            if self.msg is not None:
                msg += "\n" + self.msg
            warning_deprecated(msg)
            return fn_to_wrap(*args, **kwargs)

        return wrapped
