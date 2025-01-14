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

import types
import warnings
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast

TObj = TypeVar("TObj")


def warning_deprecated(msg: str) -> None:
    """
    Display a warning message indicating that a certain functionality is deprecated.

    :param msg: The warning message to display.
    """
    # Note: must use FutureWarning in order not to get suppressed by default
    warnings.warn(msg, FutureWarning, stacklevel=2)


def deprecated(
    msg: Optional[str] = None, start_version: Optional[str] = None, end_version: Optional[str] = None
) -> Callable[[TObj], TObj]:
    """
    Decorator to mark a function or class as deprecated.

    :param msg: Message to provide additional information about the deprecation.
    :param start_version: Start version from which the function or class is deprecated.
    :param end_version: End version until which the function or class is deprecated.

    :return: The decorator function.
    """

    def decorator(obj: TObj) -> TObj:

        if isinstance(obj, types.FunctionType):

            @wraps(obj)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                name = f"function '{obj.__module__}.{obj.__name__}'"
                text = _generate_deprecation_message(name, msg, start_version, end_version)
                warning_deprecated(text)
                return obj(*args, **kwargs)

            return cast(TObj, wrapper)

        if isinstance(obj, type):
            original_init = obj.__init__  # type: ignore[misc]

            @wraps(original_init)
            def wrapped_init(*args: Any, **kwargs: Any) -> Any:
                name = f"class '{obj.__module__}.{obj.__name__}'"
                text = _generate_deprecation_message(name, msg, start_version, end_version)
                warning_deprecated(text)
                return original_init(*args, **kwargs)

            obj.__init__ = wrapped_init  # type: ignore[misc]

            return cast(TObj, obj)

        raise TypeError("The @deprecated decorator can only be used on functions or classes.")

    return decorator


def _generate_deprecation_message(
    name: str, text: Optional[str], start_version: Optional[str], end_version: Optional[str]
) -> str:
    """
    Generate a deprecation message for a given name, with optional start and end versions.

    :param name: The name of the deprecated feature.
    :param text: Additional text to include in the deprecation message.
    :param start_version: The version from which the feature is deprecated.
    :param end_version: The version in which the feature will be removed.
    :return: The deprecation message.
    """
    msg = (
        f"Usage of {name} is deprecated {f'starting from NNCF v{start_version} ' if start_version else ''}"
        f"and will be removed in {f'NNCF v{end_version}.' if end_version else 'a future NNCF version.'}"
    )
    if text:
        return "\n".join([msg, text])
    return msg
