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
import datetime
import itertools
import os
import os.path as osp
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, List, Optional, TypeVar, Union

from tabulate import tabulate

from nncf.common.utils.os import is_windows

TKey = TypeVar("TKey", bound=Hashable)


def create_table(
    header: List[str],
    rows: List[List[Any]],
    table_fmt: str = "mixed_grid",
    max_col_widths: Optional[Union[int, Iterable[int]]] = None,
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


def configure_accuracy_aware_paths(log_dir: Union[str, Path]) -> Union[str, Path]:
    """
    Create a subdirectory inside of the passed log directory
    to save checkpoints from the accuracy-aware training loop to.

    :param log_dir: Path to the main log directory.
    :return: Path to the accuracy-aware training subdirectory.
    """
    d = datetime.datetime.now()
    run_id = "{:%Y-%m-%d__%H-%M-%S}".format(d)
    acc_aware_log_dir = osp.join(log_dir, "accuracy_aware_training/{run_id}".format(run_id=run_id))
    os.makedirs(acc_aware_log_dir, exist_ok=True)
    return acc_aware_log_dir


def product_dict(d: Dict[TKey, List[Any]]) -> Iterable[Dict[TKey, Any]]:
    """
    Generates dicts which enumerate the options for keys given in the input dict;
    options are represented by list values in the input dict.

    :param d: A list-valued dict, such as {'key': ['option1', 'option2'],
                                           'key2': ['option3', 'option4'] }
    :return: Generated dicts enumerating options, i.e.
      {'key': 'option1', 'key2': 'option3'}, {'key': 'option2', 'key2': 'option3'}, ...
    """
    keys = d.keys()
    vals = d.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))
