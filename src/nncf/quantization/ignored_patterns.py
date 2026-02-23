# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.patterns.patterns import GraphPattern


def create_rope_pattern(
    mm_metatype: OperatorMetatype,
    transpose_metatype: OperatorMetatype,
    concat_metatype: OperatorMetatype,
    cos_metatype: OperatorMetatype,
    sin_metatype: OperatorMetatype,
) -> GraphPattern:
    """
    Creates Rotational Embedding pattern.

    :param mm_metatype: MatMul metatype.
    :param transpose_metatype: Transpose metatype.
    :param concat_metatype: Concat metatype.
    :param cos_metatype: Cos metatype.
    :param sin_metatype: Sin metatype.
    :return: The Rotational Embedding pattern.
    """
    ret_pattern = GraphPattern()
    for with_transpose in [True, False]:
        pattern = GraphPattern()
        matmul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "MATMUL", GraphPattern.METATYPE_ATTR: mm_metatype})
        concat_node = pattern.add_node(
            **{GraphPattern.LABEL_ATTR: "CONCAT", GraphPattern.METATYPE_ATTR: concat_metatype}
        )
        cos_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "COS", GraphPattern.METATYPE_ATTR: cos_metatype})
        sin_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SIN", GraphPattern.METATYPE_ATTR: sin_metatype})

        if with_transpose:
            transpose_node = pattern.add_node(
                **{GraphPattern.LABEL_ATTR: "TRANSPOSE", GraphPattern.METATYPE_ATTR: transpose_metatype}
            )
            pattern.add_edge(matmul_node, transpose_node)
            pattern.add_edge(transpose_node, concat_node)
        else:
            pattern.add_edge(matmul_node, concat_node)
        pattern.add_edge(concat_node, cos_node)
        pattern.add_edge(concat_node, sin_node)
        ret_pattern.add_pattern_alternative(pattern)
    return ret_pattern
