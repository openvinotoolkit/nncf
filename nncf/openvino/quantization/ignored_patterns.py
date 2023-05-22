from nncf.common.graph.patterns.patterns import GraphPattern
from nncf.common.graph.patterns.patterns import IgnoredPatternNames
from nncf.common.utils.registry import Registry
from nncf.openvino.graph.metatypes import openvino_metatypes as om

OPENVINO_IGNORED_PATTERNS = Registry("IGNORED_PATTERNS")


@OPENVINO_IGNORED_PATTERNS.register(IgnoredPatternNames.SOFTMAX_MATMUL_ANY)
def softmax_matmul():
    pattern = GraphPattern()
    softmax = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SOFTMAX", GraphPattern.METATYPE_ATTR: om.OVSoftmaxMetatype})
    matmul = pattern.add_node(**{GraphPattern.LABEL_ATTR: "MATMUL", GraphPattern.METATYPE_ATTR: om.OVMatMulMetatype})
    any_pattern_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "ANY", GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE}
    )
    pattern.add_edge(softmax, matmul)
    pattern.add_edge(any_pattern_node, matmul)
    return pattern
