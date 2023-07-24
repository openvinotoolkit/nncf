# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import functools
import itertools
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

SearchSpace = Dict[str, Union[List[Any], "SearchSpace"]]


def is_dataclass_instance(obj: Any):
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


class ParamsTransformation:
    """
    Describes transformation to change values of parameters. The parameters
    should be represented as dictionary, where key is a name of parameter.
    """

    def __init__(self, changes: Optional[Dict[str, Any]] = None):
        """
        :param changes: Describes value changes of parameters. Should have the following
            structure of changes

                {
                    "param_name": <new_value>
                }

            where <new_value> is new value of parameter. In case when `param_name` is a dataclass
            instance it is possible to change value field of this instance. To do this the following
            structure of changes should be used

                {
                    "param_name": {
                        "field_name": <new_value>
                    }
                }
            This rule is applied to `field_name` as well and etc.
        """
        if changes is None:
            changes = {}

        self._changes = changes

    def apply(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies transformation to change value of parameters.

        :param params: Parameters to transform.
        :return: Transformed parameters.
        """
        params_copy = deepcopy(params)
        for param_name, param_value in self._changes.items():
            param = params_copy[param_name]

            if is_dataclass_instance(param) and isinstance(param_value, dict):
                params_copy[param_name] = ParamsTransformation._apply_to_dataclass_instance(param, param_value)
            else:
                params_copy[param_name] = param_value

        return params_copy

    def as_str(self) -> str:
        """
        :return: String representation.
        """
        return ParamsTransformation._as_str(self._changes)

    @classmethod
    def concatenate(
        cls, transformation_a: "ParamsTransformation", transformation_b: "ParamsTransformation"
    ) -> "ParamsTransformation":
        """
        Creates new transformation that is concatenation of `transformation_a` and `transformation_b`.

        :param transformation_a: Params transformation.
        :param transformation_b: Params transformation.
        :return: Concatenation of `transformation_a` and `transformation_b`
        """
        changes_a = deepcopy(transformation_a._changes)
        changes_b = deepcopy(transformation_b._changes)
        changes_a.update(changes_b)
        return cls(changes_a)

    @staticmethod
    def _as_str(data, indent: int = 0) -> str:
        if is_dataclass_instance(data):
            data = dataclasses.asdict(data)

        s = ""
        for param_name, param_value in data.items():
            shift = " " * indent
            s = f"{s}{shift}{param_name}: "
            if isinstance(param_value, dict) or is_dataclass_instance(param_value):
                values = ParamsTransformation._as_str(param_value, indent + 2)
                s = f"{s}\n{values}"
            else:
                s = f"{s}{param_value}\n"
        return s

    @staticmethod
    def _apply_to_dataclass_instance(instance: Any, changes: Dict[str, Any]) -> Any:
        for attr_name, changes_ in changes.items():
            attr_value = getattr(instance, attr_name)
            if is_dataclass_instance(attr_value) and isinstance(changes_, dict):
                setattr(instance, attr_name, ParamsTransformation._apply_to_dataclass_instance(attr_value, changes_))
            else:
                setattr(instance, attr_name, changes_)
        return instance


def create_params_transformation(search_space: SearchSpace) -> Dict[str, List[ParamsTransformation]]:
    """
    Creates transformations that changes value of single parameter.

    :param search_space: Describes possible values for parameters.
    :return: List of transformations.
    """
    param_name_to_transformations_map = {}
    for param_name, values in search_space.items():
        transformations = _prepare_transformations_recursively(param_name, values)
        param_name_to_transformations_map[param_name] = [ParamsTransformation(t) for t in transformations]
    return param_name_to_transformations_map


def create_combinations(
    params_transformations: Dict[str, List[ParamsTransformation]]
) -> Dict[Tuple[int, ...], ParamsTransformation]:
    """
    :param params_transformations:
    :return:
    """
    combinations = {}
    transformations = list(params_transformations.values())

    for i in range(1, len(transformations) + 1):
        for item in itertools.product(*map(enumerate, transformations[:i])):
            combination_key, combination = zip(*item)
            combination = functools.reduce(ParamsTransformation.concatenate, combination, ParamsTransformation())
            combinations[combination_key] = combination

    return combinations


def _prepare_transformations_recursively(param_name: str, values: Union[List[Any], SearchSpace]):
    if isinstance(values, list):
        transformations = [{param_name: v} for v in values]
    elif isinstance(values, dict):
        recursively_prepared = (_prepare_transformations_recursively(name, v) for name, v in values.items())
        transformations = [{param_name: v} for v in itertools.chain.from_iterable(recursively_prepared)]
    else:
        ValueError(f"Unexpected type for {param_name} parameter: {type(values)} is given but dict or list are expected")
    return transformations


def dict_product(kwargs):
    """
    Returns the cartesian product of input keyword arguments.
    Each argument should be a list.

    Creates dict from pairs `zip(kwargs.keys(), elements)` for
    each `elements` from cartesian product

        kwargs[k1] x kwargs[k2] x .. x kwargs[kn],

    where k1, k2, ..., kn from `kwargs.keys()`.

    :return: A cartesian product of input keyword arguments.
    """
    names = kwargs.keys()
    cartesian_product = [dict(zip(names, values)) for values in itertools.product(*kwargs.values())]
    return cartesian_product


def create_params(cls, **kwargs):
    return [cls(**params) for params in dict_product(kwargs)]
