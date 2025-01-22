# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

import nncf
from nncf.common.graph import NNCFNode
from nncf.experimental.torch.sparsity.movement.functions import binary_mask_by_threshold
from nncf.torch.layer_utils import COMPRESSION_MODULES
from nncf.torch.layer_utils import CompressionParameter
from nncf.torch.sparsity.functions import apply_binary_mask as apply_binary_mask_impl
from nncf.torch.sparsity.layers import BinaryMask
from nncf.torch.utils import is_tracing_state


class SparseStructure(Enum):
    FINE = "fine"
    BLOCK = "block"
    PER_DIM = "per_dim"


class SparseConfig:
    """
    Defines the sparse structure config with required options for a certain supported layer.
    """

    def __init__(
        self, mode: SparseStructure, sparse_factors: Optional[Tuple[int, int]] = None, sparse_axis: Optional[int] = None
    ):
        """
        Parses and validates the sparse structure of a certain layer for movement sparsity.

        :param mode: The sparse structure mode.
        :param sparse_factors: Block shape to sparsify as a whole in a weight. Required when `mode` is "block".
        :param sparse_axis: The dimension to sparsify in a weight. Required when `mode` is "per_dim".
        """
        error_prefix = "Invalid sparse config."
        self.sparse_factors = None
        self.sparse_axis = None
        self.mode = mode
        if self.mode == SparseStructure.FINE:
            if not (
                (isinstance(sparse_factors, (tuple, list)) and tuple(sparse_factors) == (1, 1))
                or sparse_factors is None
            ):
                raise ValueError(
                    f"{error_prefix} Fine sparse structure expects `sparse_factors` to be [1, 1] or unspecified."
                )
            if sparse_axis is not None:
                raise ValueError(f"{error_prefix} Fine sparse structure does not expect specified `axis`.")
            self.sparse_factors = (1, 1)

        if self.mode == SparseStructure.BLOCK:
            if sparse_factors is None:
                raise ValueError(
                    f"{error_prefix} Missing `sparse_factors`. Block sparsity structure expects it specified."
                )
            if not (isinstance(sparse_factors, (tuple, list)) and len(sparse_factors) == 2):
                raise ValueError(
                    f"{error_prefix} Invalid format of `sparse_factors. "
                    "Block sparsity structure expects tuple of two numbers."
                )
            if sparse_axis is not None:
                raise ValueError(f"{error_prefix} Block sparse structure does not expect specified `axis`.")
            self.sparse_factors = tuple(sparse_factors)

        if self.mode == SparseStructure.PER_DIM:
            if sparse_axis is None:
                raise ValueError(
                    f"{error_prefix} Missing `axis`. Per-dim sparsity structure expects it to be specified."
                )
            if sparse_factors is not None:
                raise ValueError(
                    f"{error_prefix} Per-dim sparsity structure does not expect specified `sparse_factors`."
                )
            self.sparse_axis = int(sparse_axis)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SparseConfig":
        """
        Creates the object from its config.

        :param config: A dict that describes the sparse structure.
        """
        mode_str = config.get("mode", SparseStructure.FINE.value)
        mode = SparseStructure(mode_str)
        sparse_factors = config.get("sparse_factors")
        axis = config.get("axis")
        return cls(mode, sparse_factors, axis)

    def __str__(self) -> str:
        return f"{self.mode.value, self.sparse_factors}"


class SparseConfigByScope:
    """
    Defines an entry for `sparse_structure_by_scopes` in movement sparsity configuration.
    It includes the sparse structure config, and the target scopes it is applied to.
    """

    def __init__(self, sparse_config: SparseConfig, target_scopes: Union[str, List[str]]):
        """
        Initializes the object that describes the sparse structure and the layers it matches.

        :param sparse_config: `SparseConfig` object that describes the sparse structure config.
        :param target_scopes: The scopes to match with this `sparse_config`.
        """
        self.sparse_config = sparse_config
        self.target_scopes = target_scopes

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SparseConfigByScope":
        """
        Creates the object from its representation.

        :param config: A dict that describes the sparse structure.
        """
        error_prefix = f"Invalid sparse structure by scopes {config}."
        target_scopes = config.get("target_scopes")
        if not target_scopes:
            raise ValueError(f"{error_prefix} Missing `target_scopes`.")
        sparse_config = SparseConfig.from_config(config)
        return cls(sparse_config, target_scopes)


@COMPRESSION_MODULES.register()
class MovementSparsifier(nn.Module):
    """
    Defines the operand of movement sparsity for supported layers.
    """

    def __init__(
        self,
        target_module_node: NNCFNode,
        sparse_cfg: SparseConfig = SparseConfig(mode=SparseStructure.FINE),
        frozen: bool = True,
        compression_lr_multiplier: Optional[float] = None,
        layerwise_loss_lambda: float = 0.5,
    ):
        """
        Initializes the operand of movement sparsity for a certain layer.

        :param target_module_node: Node name of the module this operand is related to.
        :param sparse_cfg: Sparse structure config for the module to sparsify.
        :param frozen: Whether the operand is frozen, i.e., binary masks are fixed.
        :param compression_lr_multiplier: The value of gradient multiplier for learnable parameters in the operand.
        :param layerwise_loss_lambda: The extra factor of compression loss for this specific layer.
        """
        super().__init__()
        self.target_module_node = target_module_node
        self.prune_bias = bool(target_module_node.layer_attributes.with_bias)
        self.frozen = frozen
        self.layerwise_loss_lambda = layerwise_loss_lambda
        self._importance_threshold = -math.inf
        self._importance_regularization_factor = 0.0

        weight_shape: List[int] = target_module_node.layer_attributes.get_weight_shape()
        assert len(weight_shape) == 2, "Unsupported module with weight shape not in 2D."
        self.weight_ctx = BinaryMask(weight_shape)
        self.sparse_factors = self._get_sparse_factors(weight_shape, sparse_cfg)
        self.sparse_structure = sparse_cfg.mode
        self.sparse_cfg = sparse_cfg

        weight_importance_shape = self._get_weight_importance_shape(
            weight_shape, self.sparse_factors, self.sparse_structure
        )
        self.weight_importance = CompressionParameter(
            torch.zeros(weight_importance_shape),
            requires_grad=not self.frozen,
            compression_lr_multiplier=compression_lr_multiplier,
        )
        self.weight_ctx.binary_mask = self._calc_training_binary_mask()

        if self.prune_bias:
            bias_shape = target_module_node.layer_attributes.get_bias_shape()
            self.bias_ctx = BinaryMask(bias_shape)
            bias_importance_shape = weight_importance_shape[0]
            self.bias_importance = CompressionParameter(
                torch.zeros(bias_importance_shape),
                requires_grad=not self.frozen,
                compression_lr_multiplier=compression_lr_multiplier,
            )
            self.bias_ctx.binary_mask = self._calc_training_binary_mask(is_bias=True)

        self.mask_calculation_hook = MaskCalculationHook(self)

    @property
    def importance_threshold(self):
        return self._importance_threshold

    @importance_threshold.setter
    def importance_threshold(self, value: float):
        self._importance_threshold = value

    @property
    def importance_regularization_factor(self):
        return self._importance_regularization_factor

    @importance_regularization_factor.setter
    def importance_regularization_factor(self, value: float):
        self._importance_regularization_factor = value

    def forward(
        self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if is_tracing_state():
            masked_weight = weight.mul(self.weight_ctx.binary_mask)
            masked_bias = None if bias is None else bias.mul(self.bias_ctx.binary_mask)
        else:
            weight_mask = self._calc_training_binary_mask(is_bias=False)
            masked_weight = apply_binary_mask_impl(weight_mask, weight)
            masked_bias = None
            if bias is not None:
                bias_mask = self._calc_training_binary_mask(is_bias=True)
                masked_bias = apply_binary_mask_impl(bias_mask, bias)
        return masked_weight, masked_bias

    def apply_binary_mask(self, param_tensor: torch.Tensor, is_bias: bool = False) -> torch.Tensor:
        ctx = self.bias_ctx if is_bias else self.weight_ctx
        return ctx.apply_binary_mask(param_tensor)

    def get_importance(self, is_bias: bool = False, expanded: bool = True) -> torch.Tensor:
        """
        Gets the importance score parameter of the operand.

        :param is_bias: If true, will return the bias importance. Otherwise will return the weight importance.
        :param expanded: Whether should expand the importance to the same shape as module weight or bias.
        """
        if is_bias and (not self.prune_bias):
            raise ValueError("The layer to sparsify does not contain bias.")
        importance = self.bias_importance if is_bias else self.weight_importance
        if (not expanded) or self.sparse_factors == [1, 1]:
            return importance
        expand_factors = [self.sparse_factors[0]] if is_bias else self.sparse_factors
        for dim, factor in enumerate(expand_factors):
            importance = importance.repeat_interleave(factor, dim=dim)
        return importance

    def loss(self) -> torch.Tensor:
        if self.frozen or self.importance_regularization_factor == 0.0:
            return torch.tensor(0.0, device=self._get_device())
        layer_loss = (
            torch.mean(torch.sigmoid(self.weight_importance))
            * self.layerwise_loss_lambda
            * math.prod(self.sparse_factors)
        )
        if self.prune_bias:
            layer_loss += (
                torch.mean(torch.sigmoid(self.bias_importance))
                * self.layerwise_loss_lambda
                * float(self.sparse_factors[0])
            )
        return layer_loss * self.importance_regularization_factor

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        if not requires_grad:
            self.zero_grad(set_to_none=True)  # avoid further unexpected update with Adam optimizer
        self.frozen = not requires_grad

    def extra_repr(self) -> str:
        return f"sparse_structure: {self.sparse_structure.value} {self.sparse_factors}"

    def _get_device(self) -> torch.device:
        return self.weight_importance.device

    def _calc_training_binary_mask(self, is_bias: bool = False):
        ctx = self.bias_ctx if is_bias else self.weight_ctx
        if (not self.training) or self.frozen:
            return ctx.binary_mask
        mask = binary_mask_by_threshold(
            input_tensor=self.get_importance(is_bias, expanded=True), threshold=self.importance_threshold
        )
        ctx.binary_mask = mask
        return mask

    @staticmethod
    def _get_weight_importance_shape(
        weight_shape: List[int], sparse_factors: Tuple[int, int], sparse_structure: SparseStructure
    ) -> Tuple[int, int]:
        if sparse_structure == SparseStructure.FINE:
            return weight_shape

        if sparse_structure == SparseStructure.BLOCK:
            r, c = sparse_factors
            return (weight_shape[0] // r, weight_shape[1] // c)

        if sparse_structure == SparseStructure.PER_DIM:
            score_shape = []
            for axis, (dim, factor) in enumerate(zip(weight_shape, sparse_factors)):
                assert dim % factor == 0, f"{factor} is not a factor of axis {axis} with dim size {dim}."
                score_shape.append(dim // factor)
            return tuple(score_shape)

        raise nncf.InternalError("Unknown sparse structure.")

    @staticmethod
    def _get_sparse_factors(weight_shape: List[int], sparse_config: SparseConfig) -> Tuple[int, int]:
        sparse_factors = sparse_config.sparse_factors
        if sparse_config.mode == SparseStructure.BLOCK:
            r, c = sparse_factors
            assert weight_shape[0] % r == 0, f"r: {r} is not a factor of dim axis 0."
            assert weight_shape[1] % c == 0, f"c: {c} is not a factor of dim axis 1."

        if sparse_config.mode == SparseStructure.PER_DIM:
            if sparse_config.sparse_axis < 0 or sparse_config.sparse_axis >= len(weight_shape):
                raise ValueError(
                    "Invalid axis id {}, axes range is [0, {}]".format(sparse_config.sparse_axis, len(weight_shape))
                )
            sparse_factors = deepcopy(weight_shape)
            sparse_factors[sparse_config.sparse_axis] = 1
            sparse_factors = tuple(sparse_factors)
        return sparse_factors


class MaskCalculationHook:
    """
    Hook for naming the binary masks of `MovementSparsifier` in torch model state dict.
    """

    def __init__(self, module: nn.Module):
        self.hook = module._register_state_dict_hook(self.hook_fn)

    def hook_fn(self, module, state_dict: Dict, prefix: str, local_metadata):
        state_dict[prefix + "weight_ctx._binary_mask"] = module.weight_ctx.binary_mask
        if module.prune_bias:
            state_dict[prefix + "bias_ctx._binary_mask"] = module.bias_ctx.binary_mask
        return state_dict

    def close(self):
        self.hook.remove()
