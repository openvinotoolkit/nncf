import itertools
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import DefaultDict, List
from unittest.mock import patch

import numpy as np
import onnx
import pytest
import torch
from nncf.common.sparsity.schedulers import PolynomialThresholdScheduler
from nncf.common.sparsity.statistics import MovementSparsityStatistics
from nncf.common.utils.helpers import matches_any, should_consider_scope
from nncf.torch import create_compressed_model
from nncf.torch.layer_utils import CompressionParameter
from nncf.torch.layers import NNCFLinear
from nncf.torch.module_operations import UpdateWeightAndBias
from nncf.torch.sparsity.movement.algo import (ImportanceLoss,
                                               MovementSparsifier,
                                               MovementSparsityController,
                                               SparseStructure, StructuredMask)
from onnx import numpy_helper
from pytest import approx
from tests.torch.sparsity.movement.helpers import (BaseCallback, ConfigBuilder,
                                                   bert_tiny_torch_model,
                                                   bert_tiny_unpretrained,
                                                   initialize_sparsifer_parameters,
                                                   run_movement_pipeline)
from tests.torch.test_algo_common import BasicLinearTestModel
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
    assert isinstance(compression_ctrl.scheduler, PolynomialThresholdScheduler)

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


def test_can_create_structured_masks(tmp_path):
    nncf_config = ConfigBuilder(sparse_structure_by_scopes=[]).build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_unpretrained(), nncf_config)
    assert hasattr(compression_ctrl, 'structured_ctx_by_group')
    structured_ctx_by_group = compression_ctrl.structured_ctx_by_group
    assert isinstance(structured_ctx_by_group, DefaultDict)
    assert set(structured_ctx_by_group.keys()) == set([0, 1])
    for ctx in itertools.chain(*structured_ctx_by_group.values()):
        assert isinstance(ctx, StructuredMask)
    # We currenltly do not check value correctness of `ctx.(in)dependent_mask` because they are internal variables.
    # See `test_controller_structured_mask_filling`.


def get_linear_layer_equiv_weight_bias(module: NNCFLinear):
    in_features = module.in_features
    zero_input = torch.zeros((1, in_features))
    eye_input = torch.eye(in_features)
    with torch.no_grad():
        bias = module(zero_input)
        weight = module(eye_input) - bias
    return weight.T, bias


@pytest.mark.parametrize("sparse_structure_by_scopes", [
    ["block", [2, 2], "{re}fc"],
    ["per_dim", [0], "{re}fc"],
    ["per_dim", [1], "{re}fc"],
    ["fine", [1, 1], "{re}fc"],
])
def test_layer_actual_behavior_matches_sparsifer_mask(tmp_path, sparse_structure_by_scopes):
    nncf_config = ConfigBuilder(sparse_structure_by_scopes=[sparse_structure_by_scopes]).build(
        log_dir=tmp_path, input_info=[{"sample_size": [1, 4]}])
    model = BasicLinearTestModel(size=4)
    compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)
    module_info = compression_ctrl.sparsified_module_info[0]
    operand = module_info.operand
    initialize_sparsifer_parameters(operand)
    operand.masking_threshold = 0.
    ori_weight, ori_bias = module_info.module.weight, module_info.module.bias
    masked_weight, masked_bias = operand(ori_weight, ori_bias)  # sparsifier forward function
    equiv_weight, equiv_bias = get_linear_layer_equiv_weight_bias(module_info.module)
    assert torch.allclose(equiv_weight, masked_weight)
    assert torch.allclose(equiv_bias, masked_bias)


@pytest.mark.parametrize('description', [
    # TODO: check fill operation cases
    dict(unstructured_masks=([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [0, 0, 0, 0],  # mhsa query
                             [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [1, 0, 0, 0],  # mhsa key
                             [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [0, 1, 0, 0],  # mhsa value
                             [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [0, 0, 0, 0],  # mhsa output
                             [[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]], [1, 0, 0],  # ffn intermediate
                             [[0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]], [0, 0, 0, 0]),  # ffn output
         ref_structured_masks=([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], [1, 1, 0, 0],
                               [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], [1, 1, 0, 0],
                               [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], [1, 1, 0, 0],
                               [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]], [1, 1, 1, 1],
                               [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]], [1, 1, 0],
                               [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]], [1, 1, 1, 1])),
])
def test_controller_structured_mask_filling(tmp_path, description):
    sparse_structure_by_scopes = [
        ["block", [1, 1], "{re}.*attention*"],
        ["per_dim", [0], "{re}.*BertIntermediate.*"],
        ["per_dim", [1], "{re}.*BertOutput.*"],
    ]
    nncf_config = ConfigBuilder(sparse_structure_by_scopes=sparse_structure_by_scopes).build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_unpretrained(), nncf_config)
    compressed_model.train()

    state_keys = []
    for keyword in ["attention.self.query", 'attention.self.key', 'attention.self.value',
                    'attention.output.dense', 'intermediate.dense', 'output.dense']:
        for attr in ['weight', 'bias']:
            state_keys.append(f"nncf_module.bert.encoder.layer.0.{keyword}"
                              f".pre_ops.0.op.{attr}_ctx._binary_mask")

    unstructued_state_dict = dict(zip(state_keys, map(torch.FloatTensor, description['unstructured_masks'])))
    compressed_model.load_state_dict(unstructued_state_dict, strict=False)
    compression_ctrl.reset_independent_structured_mask()
    compression_ctrl.resolve_structured_mask()
    compression_ctrl.populate_structured_mask()
    structured_state_dict = compressed_model.state_dict()
    ref_structured_state_dict = dict(zip(state_keys, map(torch.FloatTensor, description['ref_structured_masks'])))
    for key in state_keys:
        assert torch.allclose(structured_state_dict[key], ref_structured_state_dict[key])


@pytest.mark.parametrize('nncf_config_builder', [
    ConfigBuilder(),  # TODO: without stuctured_masking
    ConfigBuilder(warmup_start_epoch=0, warmup_end_epoch=1),
    ConfigBuilder(warmup_start_epoch=2, warmup_end_epoch=5),
    ConfigBuilder(warmup_start_epoch=1, warmup_end_epoch=1),
    ConfigBuilder(warmup_start_epoch=-1, warmup_end_epoch=-1),  # no warm_up?
])
def test_importance_threshold_and_regularization_factor_range(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder.build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)
    callback = BaseCallback(compression_ctrl)
    train_log, eval_result, _ = run_movement_pipeline(tmp_path, compression_ctrl, compressed_model, [callback])

    # check importance_threshold & regularization factor.
    warmup_a = nncf_config_builder.get('warmup_start_epoch')
    warmup_b = nncf_config_builder.get('warmup_end_epoch')
    importance_regularization_factor = nncf_config_builder.get('importance_regularization_factor')
    init_importance_threshold = nncf_config_builder.get('init_importance_threshold')
    final_importance_threshold = nncf_config_builder.get('final_importance_threshold')
    for epoch, log in callback.get_compress_log().items():
        # TODO: the <= and > may be a bit confusing here.
        # Assume one epoch has 4 steps, and warmup_range is [1, 2]:
        #   `epoch` here starts with [0.25, 0.5, 0.75, 1.0] for the first epoch.
        #   so the first stage is filtered by `epoch <= 1`, and the last stage is `epoch > 2`.
        if epoch <= warmup_a:
            assert log['importance_regularization_factor'] == approx(0.0)
            assert log['importance_threshold'] == approx(init_importance_threshold)
        elif epoch > warmup_b:
            assert log['importance_regularization_factor'] == approx(importance_regularization_factor)
            assert log['importance_threshold'] == approx(final_importance_threshold)
        else:
            assert 0.0 <= log['importance_regularization_factor'] <= importance_regularization_factor + 1e-6  # how to check this in Pytest?
            assert init_importance_threshold - 1e-6 <= log['importance_threshold'] <= final_importance_threshold + 1e-6  # TODO: during warmup, threshold starts at a non-inf, customizable value.


@pytest.mark.parametrize('nncf_config_builder', [
    ConfigBuilder(),
])
def test_importance_score_update(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder.build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)

    class CheckImportanceCallback(BaseCallback):
        def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
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
def test_compression_loss_update(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder.build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)

    class CheckCompressionLossCallback(BaseCallback):
        def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            super().on_step_end(args, state, control, **kwargs)
            assert isinstance(self.compression_ctrl.loss, ImportanceLoss)
            for layer in self.compression_ctrl.loss._sparse_layers:
                assert isinstance(layer, MovementSparsifier)

            # check gradient
            loss_compress = self.compression_ctrl.loss()
            assert loss_compress.requires_grad is (state.epoch <= self.compression_ctrl.scheduler.warmup_end_epoch)
            # check value
            if state.epoch <= self.compression_ctrl.scheduler.warmup_start_epoch:
                assert not torch.is_nonzero(loss_compress)
            elif self.compression_ctrl.scheduler.current_importance_lambda > 0.:  # TODO: not the right way to check condition
                assert loss_compress > 0.

    run_movement_pipeline(tmp_path, compression_ctrl, compressed_model, [CheckCompressionLossCallback(compression_ctrl)])


@pytest.mark.parametrize('nncf_config_builder', [
    ConfigBuilder(),
])
def test_binary_mask_update(tmp_path, nncf_config_builder):
    # We will check "mask == (importance_score > threshold)" in the unit test.
    nncf_config = nncf_config_builder.build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)

    class CheckBinaryMaskCallback(BaseCallback):
        def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            super().on_step_end(args, state, control, **kwargs)
            if state.epoch <= self.compression_ctrl.scheduler.warmup_start_epoch:
                for m in self.compression_ctrl.sparsified_module_info:
                    weight_mask = m.operand.weight_ctx.binary_mask
                    ref_weight_mask = torch.ones_like(m.module.weight.data)
                    assert torch.allclose(weight_mask, ref_weight_mask)
                    bias_mask = m.operand.bias_ctx.binary_mask
                    ref_bias_mask = torch.ones_like(m.module.bias.data)
                    assert torch.allclose(bias_mask, ref_bias_mask)
            elif state.epoch >= self.compression_ctrl.scheduler.warmup_end_epoch:  # when warmup is ended
                count_sparse_modules = 0
                for m in self.compression_ctrl.sparsified_module_info:
                    weight_mask = m.operand.weight_ctx.binary_mask
                    bias_mask = m.operand.bias_ctx.binary_mask
                    if torch.mean(weight_mask) < 1. or torch.mean(bias_mask) < 1.:
                        count_sparse_modules += 1
                assert count_sparse_modules > 0

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


def check_onnx_has_sparsified_param(compressed_model, compression_ctrl, onnx_path):
    compressed_model.eval()
    ref_params = {}
    module_name_dict = {module: name for name, module in compressed_model.named_modules()}
    for m in compression_ctrl.sparsified_module_info:
        with torch.no_grad():
            weight, bias = m.operand(m.module.weight, m.module.bias)
        name = module_name_dict[m.module]
        ref_params[name + '.weight'] = weight
        ref_params[name + '.bias'] = bias

    # temporary solution to preserve param names in onnx model
    with patch("torch.onnx.export", wraps=partial(torch.onnx.export, do_constant_folding=False)):
        compression_ctrl.export_model(onnx_path)
    onnx_model = onnx.load(onnx_path)

    for t in onnx_model.graph.initializer:
        if t.name in ref_params:
            ref_param = ref_params.pop(t.name).numpy()
            onnx_param = numpy_helper.to_array(t)
            assert np.allclose(ref_param, onnx_param)
    assert len(ref_params) == 0


@pytest.mark.parametrize('nncf_config_builder', [
    ConfigBuilder(),
])
def test_export_onnx_has_sparsified_param(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder.build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)
    for m in compression_ctrl.sparsified_module_info:
        initialize_sparsifer_parameters(m.operand)
    for threshold in [-0.5, 0.0, 0.5]:
        compressed_model.train()
        for m in compression_ctrl.sparsified_module_info:
            m.operand.masking_threshold = threshold
        onnx_path = tmp_path / f'model_thres{threshold}.onnx'
        check_onnx_has_sparsified_param(compressed_model, compression_ctrl, onnx_path)


@pytest.mark.parametrize('nncf_config_builder', [
    ConfigBuilder(warmup_start_epoch=0, warmup_end_epoch=1),
])
def test_structured_mask_obeys_unstructured(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder.build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)
    for m in compression_ctrl.sparsified_module_info:
        initialize_sparsifer_parameters(m.operand)

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
                unstructured_mask_dict = get_sparsified_module_mask(self.compression_ctrl)
                super().on_epoch_begin(args, state, control, **kwargs)  # conduct fill stage
                stuctured_mask_dict = get_sparsified_module_mask(self.compression_ctrl)
                for key in unstructured_mask_dict:
                    unstuctured_mask = unstructured_mask_dict[key]
                    stuctured_mask = stuctured_mask_dict[key]
                    assert torch.all(stuctured_mask >= unstuctured_mask)  # structured mask preserves more "1"s

    callback = StructuredMaskingCallback(compression_ctrl)
    run_movement_pipeline(tmp_path, compression_ctrl, compressed_model, [callback], num_train_epochs=2)


@pytest.mark.parametrize('nncf_config_builder', [
    ConfigBuilder(warmup_start_epoch=0, warmup_end_epoch=1),
])
def test_fill_stage_has_fixed_sparsity(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder.build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)
    for m in compression_ctrl.sparsified_module_info:
        initialize_sparsifer_parameters(m.operand)

    callback = BaseCallback(compression_ctrl)
    run_movement_pipeline(tmp_path, compression_ctrl, compressed_model, [callback], num_train_epochs=2)
    rela_sparsity = [log['relative_sparsity'] for epoch, log in callback.get_compress_log().items()
                     if epoch > nncf_config_builder.get('warmup_end_epoch')]
    assert all(sparsity == approx(rela_sparsity[0]) for sparsity in rela_sparsity)
