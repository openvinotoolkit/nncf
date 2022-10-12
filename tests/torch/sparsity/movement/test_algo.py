import math
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

import nncf
import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn
from nncf.common.sparsity.statistics import MovementSparsityStatistics
from nncf.common.utils.helpers import matches_any, should_consider_scope
from nncf.torch import create_compressed_model
from nncf.torch.layer_utils import CompressionParameter
from nncf.torch.layers import NNCFLinear
from nncf.torch.module_operations import UpdateWeightAndBias
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.sparsity.layers import BinaryMask
from nncf.torch.sparsity.movement.algo import (ImportanceLoss,
                                               MovementSparsifier,
                                               MovementSparsityController,
                                               SparseConfig, SparseStructure)
from onnx import numpy_helper
from pytest import approx
from tests.torch.sparsity.movement.helpers import (BaseCallback, ConfigBuilder,
                                                   bert_tiny_torch_model,
                                                   run_movement_pipeline)
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerControl, TrainerState


def check_sparsified_layer_mode(sparsifier: MovementSparsifier, module: NNCFLinear, mode, grid_size):
    weight_shape = module.weight.shape
    assert isinstance(sparsifier._weight_importance, CompressionParameter)
    if mode == SparseStructure.BLOCK:
        ref_weight_importance = torch.zeros([weight_shape[0] // grid_size[0], weight_shape[1] // grid_size[1]])
    elif mode == SparseStructure.PER_DIM:
        ref_weight_importance = torch.zeros([1, weight_shape[grid_size[0]]])
    else:
        ref_weight_importance = torch.zeros(weight_shape)
    assert torch.allclose(sparsifier._weight_importance, ref_weight_importance)  # TODO: should not use internal variables here

    if module.bias is not None:
        assert isinstance(sparsifier._bias_importance, CompressionParameter)
        ref_bias_importance = torch.zeros([ref_weight_importance.shape[0]])
        assert torch.allclose(sparsifier._bias_importance, ref_bias_importance)


@ pytest.mark.parametrize('nncf_config_builder', [
    ConfigBuilder(),  # mixed mode
    ConfigBuilder(sparse_structure_by_scopes=[]),  # implicit all fine
    ConfigBuilder(sparse_structure_by_scopes=[["fine", [1, 1], "{re}.*"]]),  # explicit all fine
    ConfigBuilder(sparse_structure_by_scopes=[["block", [8, 8], "{re}.*"]]),  # all block
    ConfigBuilder(sparse_structure_by_scopes=[["per_dim", [0], "{re}.*"]]),  # all per_dim
    ConfigBuilder(sparse_structure_by_scopes=[["per_dim", [1], "{re}.*"]]),  # all per_dim
    ConfigBuilder(sparse_structure_by_scopes=[["block", [16, 16], "{re}.*query.*"]]),  # mixed of explicit and implicit
    ConfigBuilder(sparse_structure_by_scopes=[["per_dim", [0], "{re}.*BertIntermediate.*"]]),
    ConfigBuilder(sparse_structure_by_scopes=[["per_dim", [0], "{re}.*BertIntermediate.*"],
                                              ["block", [4, 4], "{re}.*attention.*"]]),
    # MovementSparsityConfigBuilder().sparse_structure_by_scopes([["block", [5, 5, 5], "{re}.*"]]),  # wrong config
    # MovementSparsityConfigBuilder().sparse_structure_by_scopes([["block", [5, 5], "{re}.*"]]),  # wrong config
])
def test_can_create_movement_sparsity_layers(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder.build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)
    assert isinstance(compression_ctrl, MovementSparsityController)

    for scope, module in compressed_model.get_nncf_modules().items():
        if not should_consider_scope(str(scope), nncf_config_builder.get('ignored_scopes')):
            if hasattr(module, 'pre_ops'):
                for op in module.pre_ops.values():
                    assert (not isinstance(op, UpdateWeightAndBias)) or (not isinstance(op.operand, MovementSparsifier))
        else:
            count_movement_op = 0
            for op in module.pre_ops.values():
                if isinstance(op, UpdateWeightAndBias) and isinstance(op.operand, MovementSparsifier):
                    count_movement_op += 1
                    sparsifier = op.operand
                    for mode, grid_size, expression in nncf_config_builder.get('sparse_structure_by_scopes'):
                        if matches_any(str(scope), expression):
                            check_sparsified_layer_mode(sparsifier, module, mode, grid_size)
                            break  # only test the first matched expression. Need tests to confirm only one matched expression matched for each layer.
                    else:
                        check_sparsified_layer_mode(sparsifier, module, SparseStructure.FINE, (1, 1))
            assert count_movement_op == 1


@pytest.mark.parametrize('nncf_config_builder', [
    ConfigBuilder(),  # TODO: without stuctured_masking
    ConfigBuilder(warmup_start_epoch=0, warmup_end_epoch=1),
    ConfigBuilder(warmup_start_epoch=2, warmup_end_epoch=5),
    ConfigBuilder(warmup_start_epoch=1, warmup_end_epoch=1),
    ConfigBuilder(warmup_start_epoch=-1, warmup_end_epoch=-1),  # no warm_up?
])
def test_can_run_full_pipeline(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder.build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)
    callback = BaseCallback(compression_ctrl)
    train_log, eval_result, _ = run_movement_pipeline(tmp_path, compression_ctrl, compressed_model, [callback])

    # check
    warmup_a = nncf_config_builder.get('warmup_start_epoch')
    warmup_b = nncf_config_builder.get('warmup_end_epoch')
    importance_regularization_factor = nncf_config_builder.get('importance_regularization_factor')
    final_importance_threshold = nncf_config_builder.get('importance_threshold')
    for epoch, log in callback.get_compress_log().items():
        # TODO: the <= and > may be a bit confusing here.
        # Assume one epoch has 4 steps, and warmup_range is [1, 2]:
        #   `epoch` here starts with [0.25, 0.5, 0.75, 1.0] for the first epoch.
        #   so the first stage is filtered by `epoch <= 1`, and the last stage is `epoch > 2`.
        if epoch <= warmup_a:
            assert log['importance_regularization_factor'] == approx(0.0)
            assert log['importance_threshold'] == -math.inf  # TODO: this is no sigmoid
            # assert log['relative_sparsity'] = approx(0.0)
        elif epoch > warmup_b:
            assert log['importance_regularization_factor'] == approx(importance_regularization_factor)
            assert log['importance_threshold'] == approx(final_importance_threshold)
        else:
            assert 0.0 <= log['importance_regularization_factor'] <= importance_regularization_factor + 1e-6  # how to check this in Pytest?
            assert -math.inf <= log['importance_threshold'] <= final_importance_threshold + 1e-6  # TODO: during warmup, threshold starts at a non-inf, customizable value.


@pytest.mark.parametrize('nncf_config_builder', [
    ConfigBuilder(),
])
def test_importance_score_update(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder.build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)

    class CheckImportanceCallback(BaseCallback):
        def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            super().on_step_end(args, state, control, **kwargs)
            for sparse_module in self.compression_ctrl.sparsified_module_info:
                sparsifier = sparse_module.operand
                ref_requires_grad = (state.epoch <= self.compression_ctrl.scheduler.warmup_end_epoch)
                assert torch.count_nonzero(sparsifier._weight_importance) > 0  # TODO: it is not a good assertion due to randomness
                assert sparsifier._weight_importance.requires_grad is ref_requires_grad
                if sparsifier.prune_bias is not None:
                    assert torch.count_nonzero(sparsifier._bias_importance) > 0
                    assert sparsifier._bias_importance.requires_grad is ref_requires_grad

    run_movement_pipeline(tmp_path, compression_ctrl, compressed_model, [CheckImportanceCallback(compression_ctrl)])


@pytest.mark.parametrize('nncf_config_builder', [
    ConfigBuilder(),
])
def test_compression_loss(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder.build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)

    class CheckCompressionLossCallback(BaseCallback):
        def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            super().on_step_end(args, state, control, **kwargs)
            assert isinstance(self.compression_ctrl.loss, ImportanceLoss)
            for layer in self.compression_ctrl.loss._sparse_layers:
                assert isinstance(layer, MovementSparsifier)

            loss_compress = self.compression_ctrl.loss()
            if state.epoch <= self.compression_ctrl.scheduler.warmup_start_epoch:
                assert not torch.is_nonzero(loss_compress)
                assert loss_compress.requires_grad is True
            else:
                if self.compression_ctrl.scheduler.current_importance_lambda > 0.0:  # TODO: not the right way to check condition
                    assert loss_compress > 0.0
                if state.epoch > self.compression_ctrl.scheduler.warmup_end_epoch:
                    assert loss_compress.requires_grad is False

    run_movement_pipeline(tmp_path, compression_ctrl, compressed_model, [CheckCompressionLossCallback(compression_ctrl)])


@pytest.mark.parametrize('nncf_config_builder', [
    ConfigBuilder(),
])
def test_binary_mask(tmp_path, nncf_config_builder):
    # TODO: ref_binary_mask` in this test is incorrect. We have `max_percentile` argument when thresholding.
    nncf_config = nncf_config_builder.build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)

    class CheckBinaryMaskCallback(BaseCallback):
        def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            super().on_step_end(args, state, control, **kwargs)
            for sparse_module in self.compression_ctrl.sparsified_module_info:
                sparsifier = sparse_module.operand
                if state.epoch <= self.compression_ctrl.scheduler.warmup_end_epoch:  # TODO: The binary mask after fill staget is not related to importance score
                    ref_binary_mask = (sparsifier._weight_importance > self.compression_ctrl.scheduler.current_importance_threshold).float()
                    assert torch.allclose(sparsifier.weight_ctx.binary_mask, sparsifier._expand_importance(ref_binary_mask))  # TODO: add _expand_importance test in unit test.

    run_movement_pipeline(tmp_path, compression_ctrl, compressed_model, [CheckBinaryMaskCallback(compression_ctrl)])


# This part is not finished. How to check sparisty with fill stage?
@pytest.mark.parametrize('nncf_config_builder', [
    ConfigBuilder(),
])
def test_sparsity_statistics_are_increasing(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder.build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)
    assert isinstance(compression_ctrl.statistics().movement_sparsity, MovementSparsityStatistics)
    log_dict = defaultdict(list)
    compressed_model.train()
    for epoch in range(10):
        compression_ctrl.scheduler.epoch_step()
        for batch in range(4):
            compression_ctrl.scheduler.step()
            statistics = compression_ctrl.statistics().movement_sparsity
            log_dict['importance_threshold'].append(statistics.importance_threshold)
            log_dict['importance_regularization_factor'].append(statistics.importance_regularization_factor)
            log_dict['sparsity_level'].append(statistics.model_statistics.sparsity_level)
            log_dict['sparsity_level_for_layers'].append(statistics.model_statistics.sparsity_level_for_layers)

    def is_non_decreasing(x: List):
        return all(a <= b for a, b in zip(x[:-1], x[1:]))

    assert is_non_decreasing(log_dict['importance_threshold'])
    assert is_non_decreasing(log_dict['importance_regularization_factor'])
    assert is_non_decreasing(log_dict['sparsity_level'])
    assert is_non_decreasing(log_dict['sparsity_level_for_layers'])


def get_linear_weight_bias_data(module: NNCFLinear):
    # TODO: how can we get this via UpdateWeightAndBias?
    in_features = module.in_features
    zero_input = torch.zeros((1, in_features))
    eye_input = torch.eye(in_features)
    with torch.no_grad():
        bias = module(zero_input)
        weight = module(eye_input) - bias
    return weight.T, bias


def check_onnx_has_sparsified_param(compressed_model, compression_ctrl, onnx_path):
    ref_params = {}
    module_name_dict = {module: name for name, module in compressed_model.named_modules()}
    for m in compression_ctrl.sparsified_module_info:
        weight, bias = get_linear_weight_bias_data(m.module)
        name = module_name_dict[m.module]
        ref_params[name + '.weight'] = weight
        ref_params[name + '.bias'] = bias

    compression_ctrl.export_model(onnx_path)
    onnx_model = onnx.load(onnx_path)

    for t in onnx_model.graph.initializer:
        if t.name in ref_params:
            ref_param = ref_params.pop(t.name).numpy()
            onnx_param = numpy_helper.to_array(t)
            assert np.allclose(ref_param, onnx_param, atol=1e-6)
            # atol cannot be the default value, i.e., 1e-8, probably due to
            # the current way of accessing pruned reference weight and bias.
    assert len(ref_params) == 0


@pytest.mark.parametrize('nncf_config_builder', [
    ConfigBuilder(),
])
def test_export_onnx_has_sparsified_param(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder.build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)
    for m in compression_ctrl.sparsified_module_info:
        torch.nn.init.uniform_(m.operand._weight_importance, a=-1., b=1.)
        torch.nn.init.uniform_(m.operand._bias_importance, a=-1., b=1.)
    for threshold in [-0.5, 0.0, 0.5]:
        compressed_model.train()
        for m in compression_ctrl.sparsified_module_info:
            m.operand.masking_threshold = threshold
            m.operand._calc_training_binary_mask()
        onnx_path = tmp_path / f'model_thres{threshold}.onnx'
        check_onnx_has_sparsified_param(compressed_model, compression_ctrl, onnx_path)


@pytest.mark.parametrize('nncf_config_builder', [
    ConfigBuilder(warmup_start_epoch=0, warmup_end_epoch=1),
])
def test_structured_mask_obeys_unstructured(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder.build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)
    for m in compression_ctrl.sparsified_module_info:
        torch.nn.init.normal_(m.operand._weight_importance)
        torch.nn.init.normal_(m.operand._bias_importance)

    def get_sparsified_module_mask(compression_ctrl):
        mask_dict = {}
        for m in compression_ctrl.sparsified_module_info:
            mask_dict[(m, 'weight')] = m.operand.weight_ctx.binary_mask.data.clone()
            mask_dict[(m, 'bias')] = m.operand.bias_ctx.binary_mask.data.clone()
        return mask_dict

    class StructuredMaskingCallback(BaseCallback):
        def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if state.epoch != self.compression_ctrl.scheduler.warmup_end_epoch:
                super().on_epoch_begin(args, state, control, **kwargs)
            else:
                unstuctured_mask_dict = get_sparsified_module_mask(self.compression_ctrl)
                super().on_epoch_begin(args, state, control, **kwargs)  # conduct fill stage
                stuctured_mask_dict = get_sparsified_module_mask(self.compression_ctrl)
                for key in unstuctured_mask_dict:
                    unstuctured_mask = unstuctured_mask_dict[key]
                    stuctured_mask = stuctured_mask_dict[key]
                    assert torch.all(stuctured_mask >= unstuctured_mask)  # structured mask preserves more "1"s

    callback = StructuredMaskingCallback(compression_ctrl)
    run_movement_pipeline(tmp_path, compression_ctrl, compressed_model, [callback], num_train_epochs=2)
    # for epoch, log in callback.get_compress_log().items():
    #     print(log)


@pytest.mark.parametrize('nncf_config_builder', [
    ConfigBuilder(warmup_start_epoch=0, warmup_end_epoch=1),
])
def test_fill_stage_has_fixed_sparsity(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder.build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)
    for m in compression_ctrl.sparsified_module_info:
        torch.nn.init.normal_(m.operand._weight_importance)
        torch.nn.init.normal_(m.operand._bias_importance)

    callback = BaseCallback(compression_ctrl)
    run_movement_pipeline(tmp_path, compression_ctrl, compressed_model, [callback], num_train_epochs=2)
    rela_sparsity = [log['relative_sparsity'] for epoch, log in callback.get_compress_log().items()
                     if epoch > nncf_config_builder.get('warmup_end_epoch')]
    assert all(sparsity == approx(rela_sparsity[0]) for sparsity in rela_sparsity)


if __name__ == "__main__":
    # test_can_create_movement_sparsity_layers(MovementSparsityConfigBuilder())
    # test_can_run_full_pipeline(Path('/tmp'), MovementSparsityConfigBuilder)
    test_structured_mask_obeys_unstructured(Path('/tmp'), ConfigBuilder(warmup_start_epoch=0, warmup_end_epoch=1))
