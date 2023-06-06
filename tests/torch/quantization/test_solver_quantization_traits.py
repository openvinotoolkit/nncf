from nncf.common.graph import INPUT_NOOP_METATYPES
from nncf.common.graph import OUTPUT_NOOP_METATYPES
from nncf.common.quantization.quantizer_propagation.graph import QuantizerPropagationStateGraph as QPSG
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.common.quantization.quantizer_propagation.structs import QuantizerPropagationStateGraphNodeType
from nncf.torch.graph.operator_metatypes import get_operator_metatypes
from nncf.torch.quantization.default_quantization import DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT
from tests.common.quantization.mock_graphs import get_ip_graph_for_test
from tests.common.quantization.mock_graphs import get_nncf_graph_from_mock_nx_graph
from tests.common.quantization.mock_graphs import get_randomly_connected_model_graph


def test_set_quantization_traits_for_quant_prop_graph_nodes():
    # Test all patchable metatypes. If a patchable metatype is not registered
    # in quantization trait-to-metatype dict, the test will fail.
    tested_op_metatypes = get_operator_metatypes()  # type: List[Type[OperatorMetatype]]
    tested_op_names = set()
    for op_meta in tested_op_metatypes:
        if op_meta not in INPUT_NOOP_METATYPES and op_meta not in OUTPUT_NOOP_METATYPES:
            aliases = op_meta.get_all_aliases()
            for alias in aliases:
                tested_op_names.add(alias)

    # Edges should be irrelevant - using random graph
    mock_graph = get_randomly_connected_model_graph(tested_op_names)
    nncf_graph = get_nncf_graph_from_mock_nx_graph(mock_graph)
    ip_graph = get_ip_graph_for_test(nncf_graph)

    quant_prop_graph = QPSG(ip_graph)
    quant_prop_solver = QuantizerPropagationSolver(
        default_trait_to_metatype_map=DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT, run_consistency_checks=True
    )
    quant_prop_graph = quant_prop_solver.set_allowed_quantization_types_for_operator_nodes(quant_prop_graph)
    op_quant_traits_map = quant_prop_solver.get_operator_quantization_traits_map()

    for qpg_node in quant_prop_graph.nodes().values():
        if qpg_node[QPSG.NODE_TYPE_NODE_ATTR] == QuantizerPropagationStateGraphNodeType.OPERATOR:
            quant_det_id = qpg_node[QPSG.OPERATOR_METATYPE_NODE_ATTR]
            quant_types = qpg_node[QPSG.ALLOWED_INPUT_QUANTIZATION_TYPES_NODE_ATTR]
            if (
                op_quant_traits_map.get(quant_det_id, QuantizationTrait.QUANTIZATION_AGNOSTIC)
                == QuantizationTrait.INPUTS_QUANTIZABLE
            ):
                # TODO: check for correspondence of operator type and HW config to initial
                # quantization types
                assert quant_types == QuantizerPropagationSolver.DEFAULT_QUANTIZATION_TYPES


def test_quantization_traits_are_unambiguous_for_op_names():
    op_name_to_trait_dict = {}  # type: Dict[str, QuantizationTrait]
    for trait, arches in DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT.items():
        for op_meta in arches:
            aliases = op_meta.get_all_aliases()
            for alias in aliases:
                if alias in op_name_to_trait_dict:
                    assert op_name_to_trait_dict[alias] == trait
                else:
                    op_name_to_trait_dict[alias] = trait
