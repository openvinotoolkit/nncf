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

from functools import partial

from nncf.common.graph.patterns import GraphPattern
from nncf.common.utils.registry import Registry
from nncf.openvino.hardware.fused_patterns import atomic_activations_operations

AWQ_PATTERNS = Registry("awq")


@AWQ_PATTERNS.register("MatMul_Mul_MatMul")
def create_matmul_mul_matmul(matmul_metatype, multiply_metatype) -> GraphPattern:
    pattern = GraphPattern()
    linear_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: "LINEAR", GraphPattern.METATYPE_ATTR: matmul_metatype})
    mul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: multiply_metatype})
    linear_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: "LINEAR", GraphPattern.METATYPE_ATTR: matmul_metatype})

    pattern.add_edge(linear_node_1, mul_node)
    pattern.add_edge(mul_node, linear_node_2)
    return pattern


def matmul_mul_operation(matmul_metatype, multiply_metatype) -> GraphPattern:
    pattern = GraphPattern()
    linear_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "LINEAR", GraphPattern.METATYPE_ATTR: matmul_metatype})
    mul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: multiply_metatype})
    pattern.add_edge(mul_node, linear_node)

    return pattern


def matmul_operation(matmul_metatype) -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: "LINEAR", GraphPattern.METATYPE_ATTR: matmul_metatype})

    return pattern


@AWQ_PATTERNS.register("ACT_Mul_MatMul")
def create_linear_mul_activations(matmul_metatype, multiply_metatype) -> GraphPattern:
    linear = matmul_mul_operation(matmul_metatype, multiply_metatype)
    activations = atomic_activations_operations()
    activations.join_patterns(linear)
    return activations


@AWQ_PATTERNS.register("ACT_MatMul")
def create_linear_activations(matmul_metatype, _multiply_metatype) -> GraphPattern:
    linear = matmul_operation(matmul_metatype)
    activations = atomic_activations_operations()
    activations.join_patterns(linear)
    return activations


def get_awq_patterns(matmul_metatype, multiply_metatype):
    res = Registry("awq")
    for k, v in AWQ_PATTERNS.registry_dict.items():
        res.registry_dict[k] = partial(v, matmul_metatype, multiply_metatype)

    return res.registry_dict
