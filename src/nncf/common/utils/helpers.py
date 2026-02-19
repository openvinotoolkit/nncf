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
import os
from contextlib import contextmanager
from typing import Any, Hashable, Iterable, Iterator, TypeVar, Union

from tabulate import tabulate

from nncf.common.utils.os import is_windows

TKey = TypeVar("TKey", bound=Hashable)


def create_table(
    header: list[str],
    rows: list[list[Any]],
    table_fmt: str = "mixed_grid",
    max_col_widths: Union[int, Iterable[int]] | None = None,
) -> str:
    """
    Returns a string which represents a table with a header and rows.

    :param header: Table's header.
    :param rows: Table's rows.
    :param table_fmt: Type of formatting of the table.
    :param max_col_widths: Max widths of columns.
    :return: A string which represents a table with a header and rows.
    """
    if not rows:
        # For empty rows max_col_widths raises IndexError
        max_col_widths = None
    if is_windows():
        # Not all terminals on Windows supports any format of table
        table_fmt = "grid"
    return tabulate(tabular_data=rows, headers=header, tablefmt=table_fmt, maxcolwidths=max_col_widths, floatfmt=".3f")


@contextmanager
def set_env_variable(key: str, value: str) -> Iterator[None]:
    """
    Temporarily sets an environment variable.

    :param key: Environment variable name.
    :param value: Environment variable value.
    """
    old_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_value is not None:
            os.environ[key] = old_value
        else:
            del os.environ[key]
