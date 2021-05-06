"""
 Copyright (c) 2021 Intel Corporation
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
import re

from typing import List, Any
from typing import Optional
from typing import Union

from texttable import Texttable


def create_table(header: List[str], rows: List[List[Any]]) -> str:
    """
    Returns a string which represents a table with a header and rows.

    :param header: Table's header.
    :param rows: Table's rows.
    :return: A string which represents a table with a header and rows.
    """
    return Texttable().header(header).add_rows(rows, header=False).draw()


def should_consider_scope(scope_str: str, target_scopes: Optional[List[str]],
                          ignored_scopes: List[str]):
    return (target_scopes is None or in_scope_list(scope_str, target_scopes)) \
               and not in_scope_list(scope_str, ignored_scopes)


def in_scope_list(scope: str, scope_list: Union[List[str], str]) -> bool:
    if scope_list is None:
        return False

    for item in [scope_list] if isinstance(scope_list, str) else scope_list:
        if "{re}" in item:
            regex = item.replace("{re}", "")
            if re.search(regex, scope):
                return True
        else:
            if scope == item:
                return True
    return False
