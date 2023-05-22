from nncf.common.graph.patterns.patterns import GraphPattern
from nncf.common.graph.patterns.patterns import IgnoredPatternNames
from nncf.common.utils.registry import Registry
from nncf.openvino.graph.metatypes import openvino_metatypes as om

OPENVINO_IGNORED_PATTERNS = Registry("IGNORED_PATTERNS")


@OPENVINO_IGNORED_PATTERNS.register(IgnoredPatternNames.SOFTMAX_MATMUL)
def softmax_matmul():
    pattern = GraphPattern()
    softmax = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SOFTMAX", GraphPattern.METATYPE_ATTR: om.OVSoftmaxMetatype})
    matmul = pattern.add_node(**{GraphPattern.LABEL_ATTR: "MATMUL", GraphPattern.METATYPE_ATTR: om.OVMatMulMetatype})
    non_pattern_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "ANY", GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE}
    )
    pattern.add_edge(softmax, matmul)
    pattern.add_edge(non_pattern_node, matmul)
    return pattern


@OPENVINO_IGNORED_PATTERNS.register(IgnoredPatternNames.SOFTMAX_RESHAPE_MATMUL)
def softmax_reshape_matmul():
    pattern = GraphPattern()
    softmax = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SOFTMAX", GraphPattern.METATYPE_ATTR: om.OVSoftmaxMetatype})
    reshape = pattern.add_node(**{GraphPattern.LABEL_ATTR: "RESHAPE", GraphPattern.METATYPE_ATTR: om.OVReshapeMetatype})
    matmul = pattern.add_node(**{GraphPattern.LABEL_ATTR: "MATMUL", GraphPattern.METATYPE_ATTR: om.OVMatMulMetatype})
    non_pattern_node_1 = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "NON_PATTERN_1", GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE}
    )
    non_pattern_node_2 = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "NON_PATTERN_2", GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE}
    )
    pattern.add_edge(softmax, reshape)
    pattern.add_edge(non_pattern_node_1, reshape)
    pattern.add_edge(reshape, matmul)
    pattern.add_edge(non_pattern_node_2, matmul)
    return pattern
