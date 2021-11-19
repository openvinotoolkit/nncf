from collections import defaultdict
import torch
from nncf.torch.dynamic_graph.patch_pytorch import MagicFunctionsToPatch
from nncf.torch.graph.operator_metatypes import PT_OPERATOR_METATYPES
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.dynamic_graph.trace_tensor import TracedTensor
from nncf.torch.dynamic_graph.trace_tensor import TensorMeta


def test_metatype_op_names_do_not_intersect():
    function_name_to_operator_metatype = defaultdict(list)
    for operator in PT_OPERATOR_METATYPES.registry_dict:
        for function_name in PT_OPERATOR_METATYPES.get(operator).get_all_aliases():
            function_name_to_operator_metatype[function_name].append(operator)

    duplicated_operator_metatypes = {}
    for function_name, operator_metatypes in function_name_to_operator_metatype.items():
        if len(operator_metatypes) > 1:
            duplicated_operator_metatypes[function_name] = operator_metatypes
    assert not duplicated_operator_metatypes, \
        "There are duplicated function name in these metatypes: {}".format(duplicated_operator_metatypes)


def test_are_all_magic_functions_patched():
    for operator in PT_OPERATOR_METATYPES.registry_dict:
        for function_name in PT_OPERATOR_METATYPES.get(operator).get_all_aliases():
            if function_name.startswith('__') and function_name.endswith('__'):
                is_contained = False
                for _, functions in MagicFunctionsToPatch.MAGIC_FUNCTIONS_TO_PATCH.items():
                    if function_name in functions:
                        is_contained = True
                        break
                assert is_contained


def test_tensor_printing_does_not_inflate_graph():
    context_to_use = TracingContext()
    context_to_use.enable_trace_dynamic_graph()
    with context_to_use as _ctx:
        with torch.no_grad():
            tensor = torch.ones([1, 2])
            print(tensor)
            str(tensor)
            tensor.__repr__()
            tensor = TracedTensor.from_torch_tensor(tensor, TensorMeta(0, 0, tensor.shape))
            print(tensor)
            str(tensor)
            tensor.__repr__()
    assert _ctx.graph.get_nodes_count() == 0
