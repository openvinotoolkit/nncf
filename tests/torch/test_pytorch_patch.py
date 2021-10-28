import pytest
from nncf.torch.dynamic_graph.patch_pytorch import MagicFunctionsToPatch
from nncf.torch.graph.operator_metatypes import PT_OPERATOR_METATYPES


def test_is_any_metatype_op_names_duplicated():
    op_name_list = set()
    for operator in PT_OPERATOR_METATYPES.registry_dict:
        for op_name in PT_OPERATOR_METATYPES.get(operator).op_names:
            assert op_name not in op_name_list
            op_name_list.add(op_name)


def test_are_all_magic_functions_patched():
    for operator in PT_OPERATOR_METATYPES.registry_dict:
        for op_name in PT_OPERATOR_METATYPES.get(operator).op_names:
            if op_name.startswith('__') and op_name.endswith('__'):
                is_contained = False
                for namespace, functions in MagicFunctionsToPatch.MAGIC_FUNCTIONS_TO_PATCH.items():
                    if op_name in functions:
                        is_contained = True
                        break
                assert is_contained
