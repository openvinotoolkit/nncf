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
from itertools import repeat
from typing import Any, Generator, Iterable, Literal

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from nncf.common.utils.os import is_windows


def create_table(
    header: list[str],
    rows: list[list[Any]],
    table_fmt: Literal["mixed_grid", "grid"] = "mixed_grid",
    max_col_widths: int | Iterable[int] | None = None,
) -> str:
    """
    Returns a string which represents a table with a header and rows.

    :param header: Table's header.
    :param rows: Table's rows.
    :param table_fmt: Table format: "mixed_grid" (Rich's default style) or "grid" (ASCII).
    :param max_col_widths: Max widths of columns.
    :return: A string which represents a table with a header and rows.
    """
    if is_windows():
        table_fmt = "grid"
    formats = {"mixed_grid": box.HEAVY_HEAD, "grid": box.ASCII}
    if table_fmt not in formats:
        msg = f"Unsupported table format: {table_fmt}"
        raise ValueError(msg)
    table = Table(box=formats[table_fmt], show_header=bool(header), show_lines=True)
    for title in header:
        table.add_column(Text(title))
    if isinstance(max_col_widths, int):
        max_col_widths = repeat(max_col_widths)
    if max_col_widths is not None:
        for column, max_width in zip(table.columns, max_col_widths):
            column.max_width = max_width

    for row in rows:
        table.add_row(
            *(
                Text("" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value))
                for value in row
            )
        )

    console = Console(
        color_system=None,
        force_terminal=False,
        width=100_000,  # To generate table as it, no depends from terminal width
    )
    with console.capture() as capture:
        console.print(table)
    return capture.get().rstrip("\n")


@contextmanager
def set_env_variable(key: str, value: str) -> Generator[None, None, None]:
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
