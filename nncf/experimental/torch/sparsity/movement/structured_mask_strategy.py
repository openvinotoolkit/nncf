"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from abc import ABC
from abc import abstractmethod
import inspect
from typing import Dict, List, Optional, Tuple, Union

from nncf.common.utils.registry import Registry
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlockType
from nncf.torch.nncf_network import NNCFNetwork

STRUCTURED_MASK_STRATEGY = Registry('structured_mask_strategy')


def detect_supported_model_family(model: NNCFNetwork) -> Optional[str]:
    """
    Returns the model family name if the model is supported by movement sparsity to conduct structured
    masking. Such name can be used as the key in `STRUCTURED_MASK_STRATEGY` to get the corresponding
    structured mask strategy.

    :param model: The compressed model wrapped by NNCF.
    :return: The string name of the model family that `model` belongs to. If `model` is not supported,
        then returns None.
    """
    model_pymodules = inspect.getmodule(model.get_nncf_wrapped_model()).__name__.split('.')
    if len(model_pymodules) >= 3 and model_pymodules[:2] == ['transformers', 'models']:
        # the case of input model defined by HuggingFace's transformers
        model_family = f'huggingface_{model_pymodules[2]}'
        if model_family in STRUCTURED_MASK_STRATEGY.registry_dict:
            return model_family
    return None


class StructuredMaskRule:
    """
    Defines the rule to resolve the structured mask in a sparsifiable layer.
    """

    def __init__(self,
                 keywords: Union[List[str], str],
                 prune_by_row: bool,
                 prune_grid: Tuple[int, int]):
        """
        :param keywords: The patterns of module_node_name.
        :param prune_by_row: Whether the matched module should be pruned by row or column.
        :param prune_grid: The grid that should be regarded as a whole for structured mask resolution.
        """
        self.keywords: List[str] = [keywords] if isinstance(keywords, str) else keywords
        self.prune_by_row = prune_by_row
        self.prune_grid = prune_grid

    def __str__(self) -> str:
        return '{class_name}({attrs})'.format(
            class_name=self.__class__.__name__,
            attrs=', '.join(f'{k}={v}' for k, v in self.__dict__.items())
        )


class BaseStructuredMaskStrategy(ABC):
    """
    Strategy that defines how structured masking is conducted for certain
    building blocks in a model.
    """

    @classmethod
    @abstractmethod
    def from_compressed_model(cls, compressed_model: NNCFNetwork):
        """
        Initializes the structured mask strategy from the compressed model.

        :param compressed_model: The model wrapped by NNCF.
        """

    @property
    @abstractmethod
    def rules_by_group_type(self) -> Dict[BuildingBlockType, List[StructuredMaskRule]]:
        """
        Returns the rule list for each `BuildingBlockType`, which should cover all
        modules that will do structured masking as a group.

        :return: A dict specifying `StructuredMaskRule` list for each `BuildingBlockType`.
        """


class BaseTransformerStructuredMaskStrategy(BaseStructuredMaskStrategy, ABC):
    """
    Base structured mask strategy class for Transformer models.

    :param MHSA_Q: Match string for query layer in multi-head self-attention.
    :param MHSA_K: Match string for key layer in multi-head self-attention.
    :param MHSA_V: Match string for value layer in multi-head self-attention.
    :param MHSA_O: Match string for output layer in multi-head self-attention.
    :param FFN_I: Match string for intermediate layer in feed-forward network.
    :param FFN_O: Match string for output layer in feed-forward network.
    """

    MHSA_Q: str = '{re}query'
    MHSA_K: str = '{re}key'
    MHSA_V: str = '{re}value'
    MHSA_O: str = '{re}output'
    FFN_I: str = '{re}feed_forward_intermediate'
    FFN_O: str = '{re}feed_forward_output'

    def __init__(self, dim_per_head: int) -> None:
        """
        Initializes the structured mask strategy for transformer models.

        :param dim_per_head: Dimension of one attention head in transformer blocks.
        """
        super().__init__()
        self.dim_per_head = dim_per_head

    @property
    def rules_by_group_type(self) -> Dict[BuildingBlockType, List[StructuredMaskRule]]:
        config = {
            BuildingBlockType.MHSA: [
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


@STRUCTURED_MASK_STRATEGY.register('huggingface_bert')
class HuggingFaceTransformerStructuredMaskStrategy(BaseTransformerStructuredMaskStrategy):
    MHSA_Q: str = '{re}query'
    MHSA_K: str = '{re}key'
    MHSA_V: str = '{re}value'
    MHSA_O: str = '{re}BertSelfOutput'
    FFN_I: str = '{re}BertIntermediate'
    FFN_O: str = '{re}BertOutput'

    @classmethod
    def from_compressed_model(cls, compressed_model: NNCFNetwork):
        hidden_dim = compressed_model.nncf_module.bert.config.hidden_size
        num_heads = compressed_model.nncf_module.bert.config.num_attention_heads
        return cls(dim_per_head=hidden_dim // num_heads)


@STRUCTURED_MASK_STRATEGY.register('huggingface_wav2vec2')
class HuggingFaceWav2Vec2StructuredMaskStrategy(BaseTransformerStructuredMaskStrategy):
    MHSA_Q: str = '{re}q_proj'
    MHSA_K: str = '{re}k_proj'
    MHSA_V: str = '{re}v_proj'
    MHSA_O: str = '{re}out_proj'
    FFN_I: str = '{re}intermediate_dense'
    FFN_O: str = '{re}output_dense'

    @classmethod
    def from_compressed_model(cls, compressed_model: NNCFNetwork):
        hidden_dim = compressed_model.nncf_module.wav2vec2.config.hidden_size
        num_heads = compressed_model.nncf_module.wav2vec2.config.num_attention_heads
        return cls(dim_per_head=hidden_dim // num_heads)


@STRUCTURED_MASK_STRATEGY.register('huggingface_swin')
class HuggingFaceSwinStructuredMaskStrategy(BaseTransformerStructuredMaskStrategy):
    MHSA_Q: str = '{re}query'
    MHSA_K: str = '{re}key'
    MHSA_V: str = '{re}value'
    MHSA_O: str = '{re}SwinSelfOutput'
    FFN_I: str = '{re}SwinIntermediate'
    FFN_O: str = '{re}SwinOutput'

    @classmethod
    def from_compressed_model(cls, compressed_model: NNCFNetwork):
        model_config = compressed_model.nncf_module.swin.config
        dim_per_head_list = []
        for i, num_head in enumerate(model_config.num_heads):
            hidden_size = model_config.embed_dim * int(2 ** i)
            dim_per_head = hidden_size // num_head
            dim_per_head_list.append(dim_per_head)
        if any(dim != dim_per_head_list[0] for dim in dim_per_head_list[1:]):
            raise NotImplementedError('Currently we only support SwinTransformers '
                                      'whose attention heads all have the same dimension.')
        return cls(dim_per_head=dim_per_head_list[0])
