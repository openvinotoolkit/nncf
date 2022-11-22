import inspect
from typing import Dict, List, Optional, Sequence, Tuple, Union

from nncf.common.utils.registry import Registry
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlockType
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
        prune_grid: Tuple[int, int]
    ) -> None:
        self.keywords: List[str] = [keywords] if isinstance(keywords, str) else keywords
        self.prune_by_row = prune_by_row
        self.prune_grid = prune_grid

    def __str__(self) -> str:
        return "%s(%s)" % (
            self.__class__.__name__,
            ', '.join(f'{k}={v}' for k, v in self.__dict__.items())
        )


class BaseStructuredMaskStrategy:
    @property
    def strategy_by_group_type(self):
        return {}

    @classmethod
    def from_compressed_model(cls, compressed_model: NNCFNetwork):
        raise NotImplementedError()


class BaseTransformerStructuredMaskStrategy(BaseStructuredMaskStrategy):
    MHSA_Q: str = "query"
    MHSA_K: str = "key"
    MHSA_V: str = "value"
    MHSA_O: str = "output"
    FFN_I: str = "feed_forward_intermediate"
    FFN_O: str = "feed_forward_output"

    def __init__(self, dim_per_head: int) -> None:
        super().__init__()
        self.dim_per_head = dim_per_head

    @property
    def strategy_by_group_type(self) -> Dict[BuildingBlockType, List[StructuredMaskRule]]:
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


@STRUCTURED_MASK_STRATEGY.register("huggingface_bert")
class HuggingFaceTransformerStructuredMaskStrategy(BaseTransformerStructuredMaskStrategy):
    MHSA_Q: str = "query"
    MHSA_K: str = "key"
    MHSA_V: str = "value"
    MHSA_O: str = "BertSelfOutput"
    FFN_I: str = "BertIntermediate"
    FFN_O: str = "BertOutput"

    @classmethod
    def from_compressed_model(cls, compressed_model: NNCFNetwork):
        hidden_dim = compressed_model.nncf_module.bert.config.hidden_size
        num_heads = compressed_model.nncf_module.bert.config.num_attention_heads
        return cls(dim_per_head=hidden_dim // num_heads)


@STRUCTURED_MASK_STRATEGY.register("huggingface_wav2vec2")
class HuggingFaceWav2Vec2StructuredMaskStrategy(BaseTransformerStructuredMaskStrategy):
    MHSA_Q: str = "q_proj"
    MHSA_K: str = "k_proj"
    MHSA_V: str = "v_proj"
    MHSA_O: str = "out_proj"
    FFN_I: str = "intermediate_dense"
    FFN_O: str = "output_dense"

    @classmethod
    def from_compressed_model(cls, compressed_model: NNCFNetwork):
        hidden_dim = compressed_model.nncf_module.wav2vec2.config.hidden_size
        num_heads = compressed_model.nncf_module.wav2vec2.config.num_attention_heads
        return cls(dim_per_head=hidden_dim // num_heads)


@STRUCTURED_MASK_STRATEGY.register("huggingface_swin")
class HuggingFaceSwinStructuredMaskStrategy(BaseTransformerStructuredMaskStrategy):
    MHSA_Q: str = "query"
    MHSA_K: str = "key"
    MHSA_V: str = "value"
    MHSA_O: str = "SwinSelfOutput"
    FFN_I: str = "SwinIntermediate"
    FFN_O: str = "SwinOutput"

    @classmethod
    def from_compressed_model(cls, compressed_model: NNCFNetwork):
        return cls(dim_per_head=compressed_model.nncf_module.swin.config.encoder_stride)
