from nncf.common.graph import NNCFNodeExpression


class TorchHWFusedPattern:
    @staticmethod
    def get() -> NNCFNodeExpression:
        # TODO: Implement "repeating expressions" so that any number of "mergeable" operations
        # immediately following a linear/convolutional/matrix op are merged into one block
        import nncf.torch.graph.patterns as p
        full_pattern = p.LINEAR_OPS + p.ANY_BN_ACT_COMBO | p.LINEAR_OPS + p.ELTWISE_UNIFORM_OPS | \
                       p.ARITHMETIC + p.ANY_BN_ACT_COMBO | p.ANY_BN_ACT_COMBO
        return full_pattern
