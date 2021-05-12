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

from typing import List, Optional, Type

from nncf.common.graph.version_agnostic_op_names import get_version_agnostic_name
from nncf.common.utils.backend import __nncf_backend__
from nncf.common.utils.registry import Registry


class OperatorMetatype:
    """
    Base class for grouping framework operators based on their semantic meaning.
    """

    name = '' # type: str
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

        :return: A subtype list
        """
        return []


class OperatorMetatypeRegistry(Registry):
    """
    Operator Metatypes Registry
    """

    def __init__(self, name):
        """
        Initialize registry state

        :param name: The registry name
        """
        super().__init__(name)
        self._op_name_to_op_meta_dict = {}

    def register(self, name=None):
        """
        Decorator for registering operator metatypes

        :param name: The registration name
        :return: The inner function for registering operator metatypes
        """
        name_ = name
        super_register = super()._register

        def wrap(obj: Type[OperatorMetatype]):
            """
            Inner function for registering operator metatypes

            :param obj: The operator metatype
            :return: The input operator metatype
            """
            cls_name = name_
            if cls_name is None:
                cls_name = obj.__name__
            super_register(obj, cls_name)
            op_names = obj.get_all_aliases()
            for name in op_names:
                name = get_version_agnostic_name(name)
                if name not in self._op_name_to_op_meta_dict:
                    self._op_name_to_op_meta_dict[name] = obj
                else:
                    if self._op_name_to_op_meta_dict[name] != obj:
                        raise RuntimeError(
                            'Inconsistent operator metatype registry - single patched '
                            'op name maps to multiple metatypes!')
            return obj

        return wrap

    def get_operator_metatype_by_op_name(self, op_name: str) -> Type[OperatorMetatype]:
        """
        Returns the operator metatype by operator name

        :param op_name: The operator name
        :return: The operator metatype
        """
        if op_name not in self._op_name_to_op_meta_dict:
            return self._op_name_to_op_meta_dict['noop']
        return self._op_name_to_op_meta_dict[op_name]


def get_operator_metatypes() -> Optional[OperatorMetatypeRegistry]:
    """
    Returns operator metatype registry

    :return: The operator metatype registry
    """
    if __nncf_backend__ == 'Torch':
        from nncf.graph.operator_metatypes \
            import PT_OPERATOR_METATYPES
        return PT_OPERATOR_METATYPES
    if __nncf_backend__ == 'TensorFlow':
        from beta.nncf.tensorflow.graph.operator_metatypes \
            import TF_OPERATOR_METATYPES
        return TF_OPERATOR_METATYPES
    return None
