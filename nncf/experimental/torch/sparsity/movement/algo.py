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
import inspect
from copy import deepcopy
from typing import List

import torch
import torch.distributed as dist

import nncf
from nncf import NNCFConfig
from nncf.api.compression import CompressionStage
from nncf.common.accuracy_aware_training.training_loop import ADAPTIVE_COMPRESSION_CONTROLLERS
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.logging import nncf_logger
from nncf.common.scopes import matches_any
from nncf.common.sparsity.statistics import MovementSparsityStatistics
from nncf.common.statistics import NNCFStatistics
from nncf.config.extractors import extract_algo_specific_config
from nncf.experimental.torch.sparsity.movement.layers import MovementSparsifier
from nncf.experimental.torch.sparsity.movement.layers import SparseConfig
from nncf.experimental.torch.sparsity.movement.layers import SparseConfigByScope
from nncf.experimental.torch.sparsity.movement.layers import SparseStructure
from nncf.experimental.torch.sparsity.movement.loss import ImportanceLoss
from nncf.experimental.torch.sparsity.movement.scheduler import MovementPolynomialThresholdScheduler
from nncf.experimental.torch.sparsity.movement.scheduler import MovementSchedulerParams
from nncf.experimental.torch.sparsity.movement.structured_mask_handler import StructuredMaskHandler
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import TransformationPriority
from nncf.torch.layers import NNCFLinear
from nncf.torch.module_operations import UpdateWeightAndBias
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.sparsity.base_algo import BaseSparsityAlgoBuilder
from nncf.torch.sparsity.base_algo import BaseSparsityAlgoController
from nncf.torch.sparsity.base_algo import SparseModuleInfo
from nncf.torch.sparsity.collector import PTSparseModelStatisticsCollector
from nncf.torch.utils import get_model_device

SUPPORTED_NNCF_MODULES = [NNCFLinear]


@PT_COMPRESSION_ALGORITHMS.register("movement_sparsity")
class MovementSparsityBuilder(BaseSparsityAlgoBuilder):
    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)
        configs = self._algo_config.get("sparse_structure_by_scopes", [])
        self._sparse_configs_by_scopes = [SparseConfigByScope.from_config(c) for c in configs]

    def create_weight_sparsifying_operation(
        self, target_module_node: NNCFNode, compression_lr_multiplier: float
    ) -> MovementSparsifier:
        sparse_cfg = SparseConfig(SparseStructure.FINE)
        node_name = target_module_node.node_name
        matched_scopes = []
        for configs_per_scopes in self._sparse_configs_by_scopes:
            target_scopes = configs_per_scopes.target_scopes
            if matches_any(node_name, target_scopes):
                sparse_cfg = configs_per_scopes.sparse_config
                matched_scopes.append(target_scopes)
        if len(matched_scopes) >= 2:
            raise nncf.InternalError(f'"{node_name}" is matched by multiple items in `sparse_structure_by_scopes`.')

        return MovementSparsifier(
            target_module_node,
            sparse_cfg=sparse_cfg,
            frozen=False,
            compression_lr_multiplier=compression_lr_multiplier,
            layerwise_loss_lambda=0.5,
        )

    def _sparsify_weights(self, target_model: NNCFNetwork) -> List[PTInsertionCommand]:
        device = get_model_device(target_model)
        sparsified_module_nodes = target_model.nncf.get_weighted_original_graph_nodes(
            nncf_module_names=[m.__name__ for m in SUPPORTED_NNCF_MODULES]
        )
        insertion_commands = []
        for module_node in sparsified_module_nodes:
            node_name = module_node.node_name

            if not self._should_consider_scope(node_name):
                nncf_logger.info(f"Ignored adding weight sparsifier in scope: {node_name}")
                continue

            nncf_logger.debug("Adding weight sparsifier in scope: {node_name}")
            compression_lr_multiplier = self.config.get_redefinable_global_param_value_for_algo(
                "compression_lr_multiplier", self.name
            )
            sparsifying_operation = self.create_weight_sparsifying_operation(module_node, compression_lr_multiplier)
            hook = UpdateWeightAndBias(sparsifying_operation).to(device)
            insertion_commands.append(
                PTInsertionCommand(
                    PTTargetPoint(TargetType.PRE_LAYER_OPERATION, target_node_name=node_name),
                    hook,
                    TransformationPriority.SPARSIFICATION_PRIORITY,
                )
            )
            sparsified_module = target_model.nncf.get_containing_module(node_name)
            self._sparsified_module_info.append(SparseModuleInfo(node_name, sparsified_module, sparsifying_operation))

        if not insertion_commands:
            raise nncf.InternalError("No sparsifiable layer found for movement sparsity algorithm.")
        return insertion_commands

    def _build_controller(self, model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return MovementSparsityController(model, self._sparsified_module_info, self.config)


MODEL_FAMILIES = ["bert", "wav2vec2", "swin", "mobilebert", "distilbert", "clip", "vit"]


def is_supported_model_family(model: NNCFNetwork) -> None:
    """
    Checks whether the model family is supported by movement sparsity to conduct structured masking.

    :param model: The compressed model wrapped by NNCF.
    """
    model_pymodules = inspect.getmodule(model).__name__.split(".")
    is_supported = False
    if len(model_pymodules) >= 3 and model_pymodules[:2] == ["transformers", "models"]:
        # the case of input model defined by HuggingFace's transformers
        model_family = model_pymodules[2]
        is_supported = model_family in MODEL_FAMILIES
    return is_supported


@ADAPTIVE_COMPRESSION_CONTROLLERS.register("pt_movement_sparsity")
class MovementSparsityController(BaseSparsityAlgoController):
    def __init__(self, target_model: NNCFNetwork, sparsified_module_info: List[SparseModuleInfo], config: NNCFConfig):
        super().__init__(target_model, sparsified_module_info)
        algo_config = extract_algo_specific_config(config, "movement_sparsity")
        sparsify_operations = [m.operand for m in self.sparsified_module_info]
        params = deepcopy(algo_config.get("params", {}))
        self._distributed = False
        self._scheduler_params = MovementSchedulerParams.from_dict(params)
        self._scheduler = MovementPolynomialThresholdScheduler(self, self._scheduler_params)
        self._loss = ImportanceLoss(sparsify_operations)
        self._config = config

        if self._scheduler.enable_structured_masking:
            if not is_supported_model_family(self.model):
                raise nncf.UnsupportedModelError(
                    "You set `enable_structured_masking=True`, but no supported model is detected. "
                    f"Supported model families: {MODEL_FAMILIES}."
                )
            self._structured_mask_handler = StructuredMaskHandler(self.model, self.sparsified_module_info)

    @property
    def compression_rate(self) -> float:
        return self.statistics().movement_sparsity.model_statistics.sparsity_level

    def reset_independent_structured_mask(self):
        """
        Asks the structured mask handler to gather independent masks in the model.
        """
        assert self._scheduler.enable_structured_masking
        self._structured_mask_handler.update_independent_structured_mask()

    def resolve_structured_mask(self):
        """
        Asks the structured mask handler to resolve dependent masks in the model.
        """
        assert self._scheduler.enable_structured_masking
        self._structured_mask_handler.resolve_dependent_structured_mask()

    def populate_structured_mask(self):
        """
        Asks the structured mask handler to update structured binary masks in model operands.
        """
        assert self._scheduler.enable_structured_masking
        self._structured_mask_handler.populate_dependent_structured_mask_to_operand()
        self._structured_mask_handler.report_structured_sparsity(self._config.get("log_dir", "."))

    def compression_stage(self) -> CompressionStage:
        if self.scheduler.current_epoch < self._scheduler_params.warmup_start_epoch:
            return CompressionStage.UNCOMPRESSED
        if self.scheduler.current_epoch >= self._scheduler_params.warmup_end_epoch:
            return CompressionStage.FULLY_COMPRESSED
        return CompressionStage.PARTIALLY_COMPRESSED

    def distributed(self):
        if not dist.is_initialized():
            raise KeyError(
                "Could not set distributed mode for the compression algorithm "
                "because the default process group has not been initialized."
            )

        if next(self._model.parameters()).is_cuda:
            state = torch.cuda.get_rng_state()
            if dist.get_backend() == dist.Backend.NCCL:
                state = state.cuda()
            torch.distributed.broadcast(state, src=0)
            torch.cuda.set_rng_state(state.cpu())
        else:
            state = torch.get_rng_state()
            torch.distributed.broadcast(state, src=0)
            torch.set_rng_state(state)

        self._distributed = True

    def freeze(self):
        self._loss.disable()
        for minfo in self.sparsified_module_info:
            minfo.operand.requires_grad_(False)

    def statistics(self, quickly_collected_only=False) -> NNCFStatistics:
        collector = PTSparseModelStatisticsCollector(self.model, self.sparsified_module_info, supports_sparse_bias=True)
        model_statistics = collector.collect()

        stats = MovementSparsityStatistics(
            model_statistics,
            self.scheduler.current_importance_threshold,
            self.scheduler.current_importance_regularization_factor,
        )

        nncf_stats = NNCFStatistics()
        nncf_stats.register("movement_sparsity", stats)
        return nncf_stats
