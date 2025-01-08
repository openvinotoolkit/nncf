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

import operator
from typing import Callable, List, TypeVar, Union

import numpy as np

TTensor = TypeVar("TTensor")


def get_subset_indices(errors: List[float], subset_size: int) -> List[int]:
    """
    Returns `subset_size` indices of elements in the `errors` list
    that have the biggest error value. Returned indices are sorted in
    ascending order.

    :param errors: A list of errors.
    :param subset_size: A number of returned indices.
    :return: Indices of elements in the `errors` list which have the biggest error value.
    """
    ordered_indices = [idx for idx, _ in sorted(enumerate(errors), key=operator.itemgetter(1), reverse=True)]
    end_index = min(subset_size, len(ordered_indices))
    return sorted(ordered_indices[:end_index])


def get_subset_indices_pot_version(errors: List[float], subset_size: int) -> List[int]:
    """
    POT implementation of the `get_subset_indices()` method.
    """
    ordered_indices = np.flip(np.argsort(errors)).tolist()
    end_index = min(subset_size, len(ordered_indices))
    return sorted(ordered_indices[:end_index])


def select_subset(
    subset_size: int,
    reference_values_for_each_item: Union[List[float], List[List[TTensor]]],
    approximate_values_for_each_item: Union[List[float], List[List[TTensor]]],
    error_fn: Callable[[Union[float, List[TTensor]], Union[float, List[TTensor]]], float],
) -> List[int]:
    """
    Selects first `subset_size` indices of data items for which `error_fn` function gives maximal value.
    Assumes that `reference_values_for_each_item` and `approximate_values_for_each_item` lists have same
    number of items.

    :param subset_size: Number of indices that will be selected. The `len(reference_values_for_each_item)`
        indices will be selected if `subset_size` parameter is greater than the number of elements in
        the `reference_values_for_each_item`.
    :param reference_values_for_each_item: List of reference values.
    :param approximate_values_for_each_item: List of approximate values.
    :param error_fn: A function used to calculate difference between `reference_values_for_each_item[i]`
        and `approximate_values_for_each_item[i]` list.
    :return: First `subset_size` indices of data items for which `error_fn` function gives maximal value.
    """
    errors = [
        error_fn(ref_val, approx_val)
        for ref_val, approx_val in zip(reference_values_for_each_item, approximate_values_for_each_item)
    ]
    subset_indices = get_subset_indices_pot_version(errors, subset_size)

    return subset_indices
