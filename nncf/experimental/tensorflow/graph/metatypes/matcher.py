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

from typing import Type

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.experimental.tensorflow.graph.metatypes.tf_ops import TF_OPERATION_METATYPES


def get_op_metatype(op_type_name: str) -> Type[OperatorMetatype]:
    """
    Returns a metatype of the TensorFlow operation by type name.

    :param op_type_name: TensorFlow operation's type name.
    :return: A metatype.
    """
    return TF_OPERATION_METATYPES.get_operator_metatype_by_op_name(op_type_name)
