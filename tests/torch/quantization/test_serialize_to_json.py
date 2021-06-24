import json

import torch

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.structs import QuantizerConfig
from nncf.torch.dynamic_graph.context import Scope
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.torch.quantization.quantizer_setup import SingleConfigQuantizerSetup
from tests.common.serialization import check_serialization


def test_quantizer_setup_serialization():
    target_type_1 = TargetType.OPERATOR_POST_HOOK
    check_serialization(target_type_1)

    target_type_2 = TargetType.POST_LAYER_OPERATION
    check_serialization(target_type_2)

    scope = Scope.from_str('MyConv/1[2]/3[4]/5')
    assert scope == Scope.from_str(str(scope))

    pttp_1 = PTTargetPoint(target_type_1, target_node_name=str(scope), input_port_id=7)
    check_serialization(pttp_1)

    qc = QuantizerConfig()
    check_serialization(qc)

    scqp_1 = SingleConfigQuantizationPoint(pttp_1, qc, directly_quantized_operator_node_names=[str(scope)])
    check_serialization(scqp_1)

    scqs = SingleConfigQuantizerSetup()
    scqs.quantization_points = {0: scqp_1, 1: scqp_1}
    scqs.unified_scale_groups = {2: {0, 1}}
    scqs.shared_input_operation_set_groups = {2: {0, 1}}
    check_serialization(scqs)


def test_precision_float():
    f1 = [1e-3, 2e-5, 1e-35, 2e40, 1.12341e-32, 0.2341123412345, 0.542e-63]
    f1_str = json.dumps(f1)
    f1_bytes = f1_str.encode('utf-8')
    f1_t = torch.ByteTensor(list(f1_bytes))
    f2_bytes = bytes(f1_t)
    f2_str = f2_bytes.decode('utf-8')
    f2 = json.loads(f2_str)
    assert f1 == f2
