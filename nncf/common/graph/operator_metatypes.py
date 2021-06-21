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

from nncf.common.utils.backend import __nncf_backend__
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

        return any([subtype.subtype_check(metatype) for subtype in subtypes])


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
            return self._op_name_to_op_meta_dict['noop']
        return self._op_name_to_op_meta_dict[op_name]


def get_operator_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the operator metatypes.

    :return: List of operator metatypes .
    """
    if __nncf_backend__ == 'Torch':
        from nncf.torch.graph.operator_metatypes \
            import get_operator_metatypes as get_operator_metatypes_pt
        operator_metatypes = get_operator_metatypes_pt()
    elif __nncf_backend__ == 'TensorFlow':
        from nncf.tensorflow.graph.metatypes.common \
            import get_operator_metatypes as get_operator_metatypes_tf
        operator_metatypes = get_operator_metatypes_tf()
    return operator_metatypes


def get_input_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the input operator metatypes.

    :return: List of the input operator metatypes .
    """
    if __nncf_backend__ == 'Torch':
        from nncf.torch.graph.operator_metatypes \
            import get_input_metatypes as get_input_metatypes_pt
        input_metatypes = get_input_metatypes_pt()
    elif __nncf_backend__ == 'TensorFlow':
        from nncf.tensorflow.graph.metatypes.common \
            import get_input_metatypes as get_input_metatypes_tf
        input_metatypes = get_input_metatypes_tf()
    return input_metatypes


def get_output_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the output operator metatypes.

    :return: List of the output operator metatypes .
    """
    if __nncf_backend__ == 'Torch':
        from nncf.torch.graph.operator_metatypes \
            import get_output_metatypes as get_output_metatypes_pt
        output_metatypes = get_output_metatypes_pt()
    if __nncf_backend__ == 'TensorFlow':
        from nncf.tensorflow.graph.metatypes.common \
            import get_output_metatypes as get_output_metatypes_tf
        output_metatypes = get_output_metatypes_tf()
    return output_metatypes
