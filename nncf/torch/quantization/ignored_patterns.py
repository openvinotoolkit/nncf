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
from nncf.common.graph.patterns.patterns import GraphPattern
from nncf.common.graph.patterns.patterns import IgnoredPatternNames
from nncf.common.utils.registry import Registry

PT_IGNORED_PATTERNS = Registry("IGNORED_PATTERNS")


def _add_softmax_matmul(
    pattern: GraphPattern, matmul_aliases, reshape_squeeze_aliases, gather_aliases, transpose_aliases
) -> None:
    #       SOFTMAX  RESHAPE||TRANSPOSE||GATHER||SQUEEZE
    #           \              /
    #            \            /
    #             \          /
    #              \        /
    #               \      /
    #                MATMUL
    branch_matmul_nodes = reshape_squeeze_aliases + gather_aliases + transpose_aliases
    softmax = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SOFTMAX", GraphPattern.METATYPE_ATTR: "softmax"})
    matmul = pattern.add_node(**{GraphPattern.LABEL_ATTR: "MATMUL", GraphPattern.METATYPE_ATTR: matmul_aliases})
    matmul_branch_nodes = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "NON_PATTERN", GraphPattern.METATYPE_ATTR: branch_matmul_nodes}
    )
    pattern.add_edge(softmax, matmul)
    pattern.add_edge(matmul_branch_nodes, matmul)


def _add_softmax_reshape_matmul(
    pattern: GraphPattern, matmul_aliases, reshape_squeeze_aliases, gather_aliases, transpose_aliases
) -> None:
    #       SOFTMAX
    #           \
    #            \
    #             \
    #             RESHAPE   RESHAPE||TRANSPOSE||GATHER||SQUEEZE
    #                 \                 /
    #                  \               /
    #                   \             /
    #                    \           /
    #                     \         /
    #                      \       /
    #                        MATMUL
    branch_matmul_nodes = reshape_squeeze_aliases + gather_aliases + transpose_aliases
    softmax = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SOFTMAX", GraphPattern.METATYPE_ATTR: "softmax"})
    reshape = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "RESHAPE", GraphPattern.METATYPE_ATTR: reshape_squeeze_aliases}
    )
    matmul = pattern.add_node(**{GraphPattern.LABEL_ATTR: "MATMUL", GraphPattern.METATYPE_ATTR: matmul_aliases})
    matmul_branch_nodes = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "RESHAPE||TRANSPOSE||GATHER", GraphPattern.METATYPE_ATTR: branch_matmul_nodes}
    )
    pattern.add_edge(softmax, reshape)
    pattern.add_edge(reshape, matmul)
    pattern.add_edge(matmul_branch_nodes, matmul)
    return pattern


@PT_IGNORED_PATTERNS.register(IgnoredPatternNames.MULTIHEAD_ATTENTION_OUTPUT)
def create_multihead_attention_output() -> GraphPattern:
    matmul_aliases = ["linear", "addmm", "matmul", "bmm", "mm", "baddbmm"]
    reshape_squeeze_aliases = ["reshape", "view", "flatten", "squeeze", "unsqueeze", "squeeze", "flatten", "unsqueeze"]
    gather_aliases = ["gather", "index_select", "where", "index_select", "__getitem__"]
    transpose_aliases = ["transpose", "permute", "transpose_"]

    pattern = GraphPattern()
    _add_softmax_matmul(
        pattern,
        matmul_aliases=matmul_aliases,
        reshape_squeeze_aliases=reshape_squeeze_aliases,
        gather_aliases=gather_aliases,
        transpose_aliases=transpose_aliases,
    )
    _add_softmax_reshape_matmul(
        pattern,
        matmul_aliases=matmul_aliases,
        reshape_squeeze_aliases=reshape_squeeze_aliases,
        gather_aliases=gather_aliases,
        transpose_aliases=transpose_aliases,
    )
    return pattern
