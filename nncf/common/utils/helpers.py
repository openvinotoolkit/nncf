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
import itertools
import re
import os
import os.path as osp
import datetime
from typing import Dict
from typing import Hashable

from typing import List, Any
from typing import Optional
from typing import Union

from texttable import Texttable

from nncf.common.graph import NNCFNodeName
from nncf.common.quantization.structs import QuantizerId


def create_table(header: List[str], rows: List[List[Any]]) -> str:
    """
    Returns a string which represents a table with a header and rows.

    :param header: Table's header.
    :param rows: Table's rows.
    :return: A string which represents a table with a header and rows.
    """
    return Texttable().header(header).add_rows(rows, header=False).draw()


def should_consider_scope(serializable_id: Union[QuantizerId, NNCFNodeName],
                          ignored_scopes: List[str],
                          target_scopes: Optional[List[str]] = None) -> bool:
    """
    Used when an entity arising during compression has to be compared to an allowlist or a denylist of strings
    (potentially regex-enabled) that is defined in an NNCFConfig .json.

    :param serializable_id: One of the supported entity types to be matched - currently possible to pass either
    NNCFNodeName (to refer to the original model operations) or QuantizerId (to refer to specific quantizers)
    :param target_scopes: A list of strings specifying an allowlist for the serializable_id. Entries of the list
        may be prefixed with `{re}` to enable regex matching.
    :param ignored_scopes: A list of strings specifying a denylist for the serializable_id. Entries of the list
        may be prefixed with `{re}` to enable regex matching.
    :return: A boolean value specifying whether a serializable_id should be considered (i.e. "not ignored", "targeted")
    """
    string_id = str(serializable_id)
    return (target_scopes is None or matches_any(string_id, target_scopes)) \
               and not matches_any(string_id, ignored_scopes)


def matches_any(tested_str: str,
                str_or_list_to_match_to: Union[List[str], str]) -> bool:
    if str_or_list_to_match_to is None:
        return False

    str_list = [str_or_list_to_match_to] if isinstance(str_or_list_to_match_to, str) else str_or_list_to_match_to
    for item in str_list:
        if '{re}' in item:
            regex = item.replace('{re}', '')
            if re.search(regex, tested_str):
                return True
        else:
            if tested_str == item:
                return True
    return False


def configure_accuracy_aware_paths(log_dir: str) -> str:
    """
    Create a subdirectory inside of the passed log directory
    to save checkpoints from the accuracy-aware training loop to.

    :param log_dir: Path to the main log directory.
    :return: Path to the accuracy-aware training subdirectory.
    """
    d = datetime.datetime.now()
    run_id = '{:%Y-%m-%d__%H-%M-%S}'.format(d)
    acc_aware_log_dir = osp.join(log_dir,
                                 'accuracy_aware_training/{run_id}'.format(run_id=run_id))
    os.makedirs(acc_aware_log_dir, exist_ok=True)
    return acc_aware_log_dir


def product_dict(d: Dict[Hashable, List]) -> Dict:
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
