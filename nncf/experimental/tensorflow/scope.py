"""
 Copyright (c) 2022 Intel Corporation
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

from typing import Optional

import tensorflow as tf
from tensorflow.python.eager import context


def get_current_name_scope() -> str:
    """
    Returns current full name scope specified by `tf.name_scope(...)` method.

    :return: The name of scope.
    """
    if tf.executing_eagerly():
        return context.context().scope_name.rstrip('/')

    return tf.compat.v1.get_default_graph().get_name_scope()


def get_op_name(op_type_name: str, scope: Optional[str] = None) -> str:
    """
    Returns the name of operation from the current name of scope.

    :param op_type_name: Type name of operation.
    :param scope: The name of scope.
    :return: The name of operation.
    """
    if scope:
        if not scope.endswith('/'):
            return scope
        op_name = scope[:-1]
    else:
        op_name = f'{get_current_name_scope()}/{op_type_name}'

    # Remove `replica_*/` prefix from `op_name`.
    if op_name.startswith('replica'):
        op_name = op_name[op_name.find('/') + 1:]

    return op_name
