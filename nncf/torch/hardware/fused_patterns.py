from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns import HWFusedPatterns
from nncf.torch.graph.pattern_operations import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.torch.graph.pattern_operations import ARITHMETIC_OPERATIONS
from nncf.torch.graph.pattern_operations import BATCH_NORMALIZATION_OPERATIONS
from nncf.torch.graph.pattern_operations import LINEAR_OPERATIONS
from nncf.torch.graph.patterns import create_h_sigmoid_act
from nncf.torch.graph.patterns import create_h_swish_act
from nncf.torch.graph.patterns import create_swish_act


def _get_torch_hw_fused_patterns() -> HWFusedPatterns:
    retval = HWFusedPatterns()
    linear_ops = GraphPattern()
    linear_ops.add_node(**LINEAR_OPERATIONS)
    retval.register(linear_ops, LINEAR_OPERATIONS['label'], match=False)

    batch_norm = GraphPattern()
    batch_norm.add_node(**BATCH_NORMALIZATION_OPERATIONS)
    retval.register(batch_norm, BATCH_NORMALIZATION_OPERATIONS['label'], match=False)

    atomic_activations = GraphPattern()
    atomic_activations.add_node(**ATOMIC_ACTIVATIONS_OPERATIONS)
    swish = create_swish_act()
    h_sigmoid = create_h_sigmoid_act()
    h_swish = create_h_swish_act()
    activations = atomic_activations | swish | h_swish | h_sigmoid
    retval.register(activations, 'ACTIVATIONS', match=False)

    arithmetic_ops = GraphPattern()
    arithmetic_ops.add_node(**ARITHMETIC_OPERATIONS)
    retval.register(arithmetic_ops, ARITHMETIC_OPERATIONS['label'], match=False)

    batch_norm_activations_permutation = batch_norm + activations | activations + batch_norm | batch_norm | activations

    retval.register(linear_ops + batch_norm_activations_permutation, 'LINEAR + BN_ACT_PERM',
                    match=True)
    retval.register(batch_norm + activations, 'BN + ACTIVATIONS', match=True)
    retval.register(activations + batch_norm, 'ACTIVATIONS + BN', match=True)
    retval.register(arithmetic_ops + batch_norm_activations_permutation,
                    'ARITHMETIC + BN_ACT_PERM', match=True)
    return retval


PT_HW_FUSED_PATTERNS = _get_torch_hw_fused_patterns()
