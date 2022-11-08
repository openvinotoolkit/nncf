import inspect
from typing import Dict, List, Optional, Sequence, Tuple, Union

from nncf.common.utils.registry import Registry
from nncf.experimental.torch.search_building_blocks.search_blocks import \
    BuildingBlockType
from nncf.torch.nncf_network import NNCFNetwork

STRUCTURED_MASK_STRATEGY = Registry("structured_mask_strategy")


def detect_supported_model_family(model):
    # TODO: review and discuss
    model_pymodules = inspect.getmodule(model.get_nncf_wrapped_model()).__name__.split(".")
    if len(model_pymodules) >= 3 and model_pymodules[:2] == ['transformers', 'models']:
        # the case of input model defined by HuggingFace's transformers
        model_family = f'huggingface_{model_pymodules[2]}'
        if model_family in STRUCTURED_MASK_STRATEGY.registry_dict:
            return model_family
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

    def __init__(self, dim_per_head: int) -> None:
        super().__init__()
        self.dim_per_head = dim_per_head

    @classmethod
    def from_compressed_model(cls, compressed_model: NNCFNetwork):
        hidden_dim = compressed_model.nncf_module.bert.config.hidden_size
        num_heads = compressed_model.nncf_module.bert.config.num_attention_heads
        return cls(dim_per_head=hidden_dim // num_heads)

    @property
    def strategy_by_group_type(self) -> Dict[str, List[StructuredMaskRule]]:
        config = {
            BuildingBlockType.MSHA: [
                StructuredMaskRule(
                    keywords=[self.MHSA_Q, self.MHSA_K, self.MHSA_V],
                    prune_by_row=True,
                    prune_grid=(self.dim_per_head, -1),
                ),
                StructuredMaskRule(
                    keywords=[self.MHSA_O],
                    prune_by_row=False,
                    prune_grid=(-1, self.dim_per_head),
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

@STRUCTURED_MASK_STRATEGY.register("huggingface_wav2vec2")
class HuggingFaceWav2Vec2StructuredMaskStrategy(HuggingFaceBertStructuredMaskStrategy):
    MHSA_Q: str = "q_proj"
    MHSA_K: str = "k_proj"
    MHSA_V: str = "v_proj"
    MHSA_O: str = "out_proj"
    FFN_I: str = "intermediate_dense"
    FFN_O: str = "output_dense"


@STRUCTURED_MASK_STRATEGY.register("huggingface_swin")
class HuggingFaceSwinStructuredMaskStrategy(BaseStructuredMaskStrategy):
    MHSA_Q: str = "query"
    MHSA_K: str = "key"
    MHSA_V: str = "value"
    MHSA_O: str = "SwinSelfOutput"
    FFN_I: str = "SwinIntermediate"
    FFN_O: str = "SwinOutput"

    # Description of config in HuggingFace's Swin
    # swin.config.depth: a list specifiying the number of Swin Transformer Block in each stage. 
    #                    E.g. depth of Swin-b is [2, 2, 18, 2], meaning there 4 stages.
    #
    # swin.config.num_heads: a list of specifiying the number of self-attention head in each stage/
    #                        E.g. Swin-b has [4, 8, 16, 32]. Together with depth config, stage 3 has 18 Swin Transformer Blocks,
    #                        each Swin transformer block in this stage has 16 attention heads.
    #
    # swin.config.encoder_stride: synonymous to the number of output dimension in each self-attention head, dim_per_head
    
    def __init__(self, dim_per_head: int) -> None:
        super().__init__()
        self.dim_per_head = dim_per_head

    @classmethod
    def from_compressed_model(cls, compressed_model: NNCFNetwork):
        return cls(
            dim_per_head=compressed_model.nncf_module.swin.config.encoder_stride
        )

    @property
    def strategy_by_group_type(self) -> Dict[str, List[StructuredMaskRule]]:
        config = {
            BuildingBlockType.MSHA: [
                StructuredMaskRule(
                    keywords=[self.MHSA_Q, self.MHSA_K, self.MHSA_V],
                    prune_by_row=True,
                    prune_grid=(self.dim_per_head, -1),
                ),
                StructuredMaskRule(
                    keywords=[self.MHSA_O],
                    prune_by_row=False,
                    prune_grid=(-1, self.dim_per_head),
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
