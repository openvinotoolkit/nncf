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

from typing import List
from typing import Optional
from typing import Type

from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.utils.registry import Registry


class OperatorMetatype:
    """
    Base class for grouping framework operators based on their semantic meaning.
    """

    name = ''  # type: str
    hw_config_names = []  # type: List[str]

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        """
        Returns a list of the framework operator aliases.

        :return: A list of the framework operator aliases.
        """
        return []

    @classmethod
    def get_subtypes(cls) -> List[Type['OperatorMetatype']]:
        """
        Returns a list of 'OperatorMetatype' that are subtypes.

        :return: A subtype list.
        """
        return []

    @classmethod
    def subtype_check(cls, metatype: Type['OperatorMetatype']) -> bool:
        """
        Check if a metatype is a subtype.

        :param metatype: An operator metatype.
        :return: True if metatype is a subtype otherwise False
        """
        subtypes = cls.get_subtypes()
        if metatype == cls or metatype in subtypes:
            return True

        return any(subtype.subtype_check(metatype) for subtype in subtypes)


class OperatorMetatypeRegistry(Registry):
    """
    Operator Metatypes Registry.
    """

    def __init__(self, name: str):
        """
        Initialize registry state.

        :param name: The registry name.
        """
        super().__init__(name)
        self._op_name_to_op_meta_dict = {}

    def register(self, name: Optional[str] = None):
        """
        Decorator for registering operator metatypes.

        :param name: The registration name.
        :return: The inner function for registering operator metatypes.
        """
        name_ = name
        super_register = super()._register

        def wrap(obj: Type[OperatorMetatype]):
            """
            Inner function for registering operator metatypes.

            :param obj: The operator metatype.
            :return: The input operator metatype.
            """
            cls_name = name_
            if cls_name is None:
                cls_name = obj.__name__
            super_register(obj, cls_name)
            op_names = obj.get_all_aliases()
            for name in op_names:
                if name in self._op_name_to_op_meta_dict \
                        and not obj.subtype_check(self._op_name_to_op_meta_dict[name]):
                    raise RuntimeError(
                        'Inconsistent operator metatype registry - single patched '
                        'op name maps to multiple metatypes!')

                self._op_name_to_op_meta_dict[name] = obj
            return obj

        return wrap

    def get_operator_metatype_by_op_name(self, op_name: str) -> Type[OperatorMetatype]:
        """
        Returns the operator metatype by operator name.

        :param op_name: The operator name.
        :return: The operator metatype.
        """
        if op_name not in self._op_name_to_op_meta_dict:
            return UnknownMetatype
        return self._op_name_to_op_meta_dict[op_name]


NOOP_METATYPES = Registry('noop_metatypes')
INPUT_NOOP_METATYPES = Registry('input_noop_metatypes')
OUTPUT_NOOP_METATYPES = Registry('output_noop_metatypes')


class UnknownMetatype(OperatorMetatype):
    """
    UnknownMetatype is mapped to operations in NNCFGraph, which are unknown for algorithms,
    typically these are the operations that haven't been discovered before.
    Algorithms should avoid processing graph nodes with this metatype.
    """
    name = "unknown"

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return [cls.name]


@NOOP_METATYPES.register()
class NoopMetatype(OperatorMetatype):
    """
    NoopMetatype is mapped to operations in NNCFGraph, that doesn't influence an input tensor.
    The compression algorithms can safely ignore this node.
    """
    name = "noop"

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return [cls.name]


@NOOP_METATYPES.register()
@INPUT_NOOP_METATYPES.register()
class InputNoopMetatype(OperatorMetatype):
    name = "input_noop"

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return [NNCFGraphNodeType.INPUT_NODE]


@NOOP_METATYPES.register()
@OUTPUT_NOOP_METATYPES.register()
class OutputNoopMetatype(OperatorMetatype):
    name = "output_noop"

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return [NNCFGraphNodeType.OUTPUT_NODE]
