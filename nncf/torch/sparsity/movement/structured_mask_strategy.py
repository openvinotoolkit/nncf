from typing import Dict, List, Optional, Sequence, Tuple, Union

from nncf.common.utils.registry import Registry
from nncf.experimental.torch.search_building_blocks.search_blocks import \
    BuildingBlockType
from nncf.torch.nncf_network import NNCFNetwork

STRUCTURED_MASK_STRATEGY = Registry("structured_mask_strategy")

def detect_supported_model_family(model):
    # TODO: need discussion on how to implement this
    for name, _ in model.named_modules():
        if 'bert' in name.lower():
            return 'huggingface_bert'
    return None
    
class StructuredMaskRule:
    def __init__(
        self,
        keywords: Union[List[str], str],
        prune_by_row: bool,
        prune_grid: Tuple[int, int],
        rule_name: Optional[str] = None,
        binary_mask_slice: Union[Tuple[slice, slice], Tuple[slice]] = (
            slice(None),
            slice(None),
        ),
    ) -> None:
        self.keywords: List[str] = [keywords] if isinstance(keywords, str) else keywords
        self.prune_by_row = prune_by_row
        self.prune_grid = prune_grid
        self.rule_name = "undefined" if rule_name is None else str(rule_name)
        self.binary_mask_slice = binary_mask_slice

    def __repr__(self) -> str:
        return "<%s \"%s\" with config=\"%r\">" % (
            self.__class__.__name__,
            self.rule_name,
            self.__dict__,
        )


class BaseStructuredMaskStrategy:
    @property
    def strategy_by_group_type(self):
        pass

    @classmethod
    def from_compressed_model(cls, compressed_model: NNCFNetwork):
        raise NotImplementedError()


@STRUCTURED_MASK_STRATEGY.register("huggingface_bert")
class HuggingFaceBertStructuredMaskStrategy(BaseStructuredMaskStrategy):
    MHSA_Q: str = "query"
    MHSA_K: str = "key"
    MHSA_V: str = "value"
    MHSA_O: str = "BertSelfOutput"
    FFN_I: str = "BertIntermediate"
    FFN_O: str = "BertOutput"

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    @classmethod
    def from_compressed_model(cls, compressed_model: NNCFNetwork):
        return cls(
            hidden_dim=compressed_model.nncf_module.bert.config.hidden_size,
            num_heads=compressed_model.nncf_module.bert.config.num_attention_heads,
        )

    @property
    def strategy_by_group_type(self) -> Dict[str, List[StructuredMaskRule]]:
        config = {
            BuildingBlockType.MSHA: [
                StructuredMaskRule(
                    keywords=[self.MHSA_Q, self.MHSA_K, self.MHSA_V],
                    prune_by_row=True,
                    prune_grid=(self.hidden_dim // self.num_heads, -1),
                ),
                StructuredMaskRule(
                    keywords=[self.MHSA_O],
                    prune_by_row=False,
                    prune_grid=(-1, self.hidden_dim // self.num_heads),
                ),
            ],
            BuildingBlockType.FF: [
                StructuredMaskRule(
                    keywords=[self.FFN_I],
                    prune_by_row=True,
                    prune_grid=(1, -1),
                ),
                StructuredMaskRule(
                    keywords=[self.FFN_O],
                    prune_by_row=False,
                    prune_grid=(-1, 1),
                ),
            ],
        }
        return config


# TODO: not tested msft swin. The BuildingBlockType may not work
@STRUCTURED_MASK_STRATEGY.register("microsoft_swin")
class MSFTSwinStructuredMaskStrategy(BaseStructuredMaskStrategy):
    MHSA_QKV: str = "qkv"
    MHSA_O: str = "output"
    FFN_I: str = "fc1"
    FFN_O: str = "fc2"

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    @property
    def strategy_by_group_type(self) -> Dict[str, List[StructuredMaskRule]]:
        head_dim = self.hidden_dim // self.num_heads
        config = {
            BuildingBlockType.MSHA: [
                StructuredMaskRule(
                    keywords=[self.MHSA_QKV],
                    prune_by_row=True,
                    prune_grid=(head_dim, -1),
                    binary_mask_slice=(slice(head_dim * i, head_dim * (i + 1)),),
                )
                for i in range(3)
            ]
            + [
                StructuredMaskRule(
                    keywords=[self.MHSA_O],
                    prune_by_row=False,
                    prune_grid=(-1, head_dim),
                )
            ],
            BuildingBlockType.FF: [
                StructuredMaskRule(
                    keywords=[self.FFN_I],
                    prune_by_row=True,
                    prune_grid=(1, -1),
                ),
                StructuredMaskRule(
                    keywords=[self.FFN_O],
                    prune_by_row=False,
                    prune_grid=(-1, 1),
                ),
            ],
        }
        return config
