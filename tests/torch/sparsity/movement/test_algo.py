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
from datasets import load_dataset
from nncf import NNCFConfig
from nncf.api.compression import CompressionAlgorithmController
from nncf.common.sparsity.statistics import MovementSparsityStatistics
from nncf.common.utils.helpers import matches_any, should_consider_scope
from nncf.torch import create_compressed_model, register_default_init_args
from nncf.torch.layer_utils import COMPRESSION_MODULES, CompressionParameter
from nncf.torch.layers import NNCF_MODULES_MAP
from nncf.torch.module_operations import UpdateWeightAndBias
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.sparsity.functions import apply_binary_mask
from nncf.torch.sparsity.layers import BinaryMask
from nncf.torch.sparsity.movement.algo import (ImportanceLoss,
                                               MovementSparsifier,
                                               MovementSparsityController,
                                               SparseConfig, SparseStructure)
from pytest import approx
from tests.torch.helpers import (BasicConvTestModel, MockModel,
                                 check_correct_nncf_modules_replacement,
                                 create_compressed_model_and_algo_for_test,
                                 get_empty_config)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
from transformers.trainer_callback import (TrainerCallback, TrainerControl,
                                           TrainerState)

print('hello?')
MODEL_NAME_HF = "google/bert_uncased_L-2_H-128_A-2"


class MovementSparsityConfigBuilder:

    def __init__(self) -> None:
        self._config_dict = {
            "input_info": [
                {"sample_size": [1, 256], "type": "long", "keyword": "input_ids"},
                {"sample_size": [1, 256], "type": "long", "keyword": "token_type_ids"},
                {"sample_size": [1, 256], "type": "long", "keyword": "position_ids"},
                {"sample_size": [1, 256], "type": "long", "keyword": "attention_mask"},
            ],
            "compression":
                {
                    "algorithm": "movement_sparsity",
                    "params": {
                        "schedule": "threshold_polynomial_decay",
                        "power": 3,
                        "init_importance_threshold": -0.1,
                        "final_importance_threshold": 0.0,
                        "warmup_start_epoch": 1,
                        "warmup_end_epoch": 3,
                        "steps_per_epoch": 128 // 32,
                        "importance_regularization_factor": 0.02,
                        "update_per_optimizer_step": True,
                    },
                    "sparse_structure_by_scopes": [
                        ["block", [8, 8], "{re}.*attention*"],
                        ["per_dim", [0], "{re}.*BertIntermediate.*"],
                        ["per_dim", [1], "{re}.*BertOutput.*"],
                    ],
                    "ignored_scopes": [
                        "{re}embedding", "{re}pooler", "{re}classifier"],
            },
        }

    def __call__(self, *args, **kwargs):
        return self

    def warmup_range(self, a=None, b=None):
        if a is not None:
            self._config_dict['compression']['params']['warmup_start_epoch'] = a
        if b is not None:
            self._config_dict['compression']['params']['warmup_end_epoch'] = b
        return self

    def input_info(self, value):
        self._config_dict['input_info'] = value
        return self

    def sparse_structure_by_scopes(self, value):
        self._config_dict['compression']['sparse_structure_by_scopes'] = value
        return self

    def ignored_scopes(self, value):
        self._config_dict['compression']['ignored_scopes'] = value
        return self

    def build(self):
        return NNCFConfig.from_dict(self._config_dict)

    def get_ignored_scopes(self):
        return self._config_dict['compression']['ignored_scopes'].copy()

    def get_sparse_structure_by_scopes(self):
        return self._config_dict['compression']['sparse_structure_by_scopes'].copy()

    def get_warmup_range(self):
        return (self._config_dict['compression']['params']['warmup_start_epoch'],
                self._config_dict['compression']['params']['warmup_end_epoch'])

    def get_final_importance_threshold(self):
        return self._config_dict['compression']['params']['final_importance_threshold']

    def get_importance_regularization_factor(self):
        return self._config_dict['compression']['params']['importance_regularization_factor']

# @pytest.fixture


def yelp_dataset():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_HF)
    dataset = load_dataset("yelp_review_full")
    max_length = 128
    n_train_samples, n_val_samples = 128, 128

    def tokenize_fn(examples):
        row = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
        row['position_ids'] = list(range(max_length))
        return row

    train_dataset = dataset["train"].shuffle(seed=42).select(range(n_train_samples)).map(tokenize_fn)
    test_dataset = dataset["test"].shuffle(seed=42).select(range(n_val_samples)).map(tokenize_fn)
    return train_dataset, test_dataset


# @pytest.fixture
def bert_tiny_torch_model():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_HF, num_labels=5)


class MovementSparsityTrainer(Trainer):
    def __init__(self, compression_ctrl, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compression_ctrl = compression_ctrl

    def compute_loss(self, model, inputs, return_outputs=False):
        loss_main, outputs = super().compute_loss(model, inputs, return_outputs=True)
        loss_compress = self.compression_ctrl.loss()
        loss = loss_main + loss_compress
        return (loss, outputs) if return_outputs else loss


class MovementSparsityCallback(TrainerCallback):
    def __init__(self, compression_ctrl: CompressionAlgorithmController) -> None:
        self.compression_ctrl = compression_ctrl
        self._compress_log = dict()

    def get_compress_log(self):
        return self._compress_log.copy()

    def get_train_log(self):
        return self._train_log.copy()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._train_log = state.log_history

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.compression_ctrl.scheduler.step()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        movement_ctrl_statistics = self.compression_ctrl.statistics().movement_sparsity
        info = dict(
            step=state.global_step,
            epoch=state.epoch,
            importance_regularization_factor=movement_ctrl_statistics.importance_regularization_factor,
            importance_threshold=movement_ctrl_statistics.importance_threshold,
            relative_sparsity=movement_ctrl_statistics.model_statistics.sparsity_level_for_layers,
        )
        self._compress_log[float(state.epoch)] = info

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.compression_ctrl.scheduler.epoch_step()
        mvmt_ctrl = self.compression_ctrl  # TODO: mutliple child ctrls
        # should move into the scheduler codes
        if self.compression_ctrl.scheduler.get_state()["current_epoch"] == mvmt_ctrl.scheduler.warmup_end_epoch:
            mvmt_ctrl.reset_independent_structured_mask()
            mvmt_ctrl.resolve_structured_mask()
            mvmt_ctrl.populate_structured_mask()


def assert_sparsified_layer_mode(sparsifier: MovementSparsifier, module: nn.Linear, mode, grid_size):
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


@pytest.mark.parametrize('nncf_config_builder', [
    MovementSparsityConfigBuilder(),  # mixed mode
    MovementSparsityConfigBuilder().sparse_structure_by_scopes([]),  # implicit all fine
    MovementSparsityConfigBuilder().sparse_structure_by_scopes([["fine", [1, 1], "{re}.*"]]),  # explicit all fine
    MovementSparsityConfigBuilder().sparse_structure_by_scopes([["block", [8, 8], "{re}.*"]]),  # all block
    MovementSparsityConfigBuilder().sparse_structure_by_scopes([["per_dim", [0], "{re}.*"]]),  # all per_dim
    MovementSparsityConfigBuilder().sparse_structure_by_scopes([["per_dim", [1], "{re}.*"]]),  # all per_dim
    MovementSparsityConfigBuilder().sparse_structure_by_scopes([["block", [16, 16], "{re}.*query.*"]]),  # mixed of explicit and implicit
    MovementSparsityConfigBuilder().sparse_structure_by_scopes([["per_dim", [0], "{re}.*BertIntermediate.*"]]),
    MovementSparsityConfigBuilder().sparse_structure_by_scopes([["per_dim", [0], "{re}.*BertIntermediate.*"],
                                                                ["block", [4, 4], "{re}.*attention.*"]]),
    # MovementSparsityConfigBuilder().sparse_structure_by_scopes([["block", [5, 5, 5], "{re}.*"]]),  # wrong config
    # MovementSparsityConfigBuilder().sparse_structure_by_scopes([["block", [5, 5], "{re}.*"]]),  # wrong config
])
def test_can_create_movement_sparsity_layers(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder.build()
    nncf_config['log_dir'] = tmp_path
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)
    assert isinstance(compression_ctrl, MovementSparsityController)

    for scope, module in compressed_model.get_nncf_modules().items():
        if not should_consider_scope(str(scope), nncf_config_builder.get_ignored_scopes()):
            if hasattr(module, 'pre_ops'):
                for op in module.pre_ops.values():
                    assert (not isinstance(op, UpdateWeightAndBias)) or (not isinstance(op.operand, MovementSparsifier))
        else:
            count_movement_op = 0
            for op in module.pre_ops.values():
                if isinstance(op, UpdateWeightAndBias) and isinstance(op.operand, MovementSparsifier):
                    count_movement_op += 1
                    sparsifier = op.operand
                    for mode, grid_size, expression in nncf_config_builder.get_sparse_structure_by_scopes():
                        if matches_any(str(scope), expression):
                            assert_sparsified_layer_mode(sparsifier, module, mode, grid_size)
                            break  # only test the first matched expression. Need tests to confirm only one matched expression matched for each layer.
                    else:
                        assert_sparsified_layer_mode(sparsifier, module, SparseStructure.FINE, (1, 1))
            assert count_movement_op == 1


def run_movement_pipeline(tmp_path, compression_ctrl, compressed_model, callbacks: Optional[List[TrainerCallback]] = None):  # temporarily remove fixture for model & dataset
    train_dataset, eval_dataset = yelp_dataset()

    training_args = TrainingArguments(
        output_dir=tmp_path / "test_trainer",
        label_names=["labels"],
        evaluation_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=8,
        learning_rate=1e-3,
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to=None,
        disable_tqdm=True,
        no_cuda=True,  # TODO: where to set cuda devices for cuda training?
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return dict(acc=(predictions == labels).mean())

    if callbacks is None:
        callbacks = [MovementSparsityCallback(compression_ctrl)]
    trainer = MovementSparsityTrainer(
        model=compressed_model,
        args=training_args,
        compression_ctrl=compression_ctrl,
        callbacks=callbacks,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()
    return train_result, eval_result, callbacks


@pytest.mark.parametrize('nncf_config_builder', [
    MovementSparsityConfigBuilder(),
    MovementSparsityConfigBuilder(),  # TODO: without stuctured_masking
    MovementSparsityConfigBuilder().warmup_range(0, 1),
    MovementSparsityConfigBuilder().warmup_range(2, 10),
    MovementSparsityConfigBuilder().warmup_range(1, 1),
    MovementSparsityConfigBuilder().warmup_range(-1, -1),  # no warm_up?
    MovementSparsityConfigBuilder().warmup_range(2, 100)]
)
def test_can_run_full_pipeline(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder().build()
    nncf_config['log_dir'] = tmp_path
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)
    callback = MovementSparsityCallback(compression_ctrl)
    train_log, eval_result, _ = run_movement_pipeline(tmp_path, compression_ctrl, compressed_model, [callback])

    # check
    warmup_a, warmup_b = nncf_config_builder().get_warmup_range()
    importance_regularization_factor = nncf_config_builder().get_importance_regularization_factor()
    final_importance_threshold = nncf_config_builder().get_final_importance_threshold()
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
    MovementSparsityConfigBuilder()]
)
def test_importance_score_update(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder().build()
    nncf_config['log_dir'] = tmp_path
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)

    class CheckImportanceCallback(MovementSparsityCallback):
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
    MovementSparsityConfigBuilder(),
])
def test_compression_loss(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder().build()
    nncf_config['log_dir'] = tmp_path
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)

    class CheckCompressionLossCallback(MovementSparsityCallback):
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
    MovementSparsityConfigBuilder(),
])
def test_binary_mask(tmp_path, nncf_config_builder):
    nncf_config = nncf_config_builder().build()
    nncf_config['log_dir'] = tmp_path
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_torch_model(), nncf_config)

    class CheckBinaryMask(MovementSparsityCallback):
        def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            super().on_step_end(args, state, control, **kwargs)
            for sparse_module in self.compression_ctrl.sparsified_module_info:
                sparsifier = sparse_module.operand
                if state.epoch <= self.compression_ctrl.scheduler.warmup_end_epoch:  # TODO: The binary mask after fill staget is not related to importance score
                    ref_binary_mask = (sparsifier._weight_importance > self.compression_ctrl.scheduler.current_importance_threshold).float()
                    torch.allclose(sparsifier.weight_ctx.binary_mask, sparsifier._expand_importance(ref_binary_mask))  # TODO: add _expand_importance test in unit test.

    run_movement_pipeline(tmp_path, compression_ctrl, compressed_model, [CheckBinaryMask(compression_ctrl)])


# @pytest.mark.parametrize('config', [
#     dict(weight=torch.ones([4, 2]),
#          bias=torch.ones([4]),
#          weight_importance=torch.tensor([[1.0], [0.0]]),
#          bias_importance=torch.tensor([1.0, 0.0]),
#          ref_sparsity=0.5,
#          nncf_config=MovementSparsityConfigBuilder().sparse_structure_by_scopes([["block", [2, 2], "{re}.*"]])),
#     dict(weight=torch.zeros([4, 2]),
#          bias=torch.ones([4]),
#          weight_importance=torch.tensor([[1.0], [1.0]]),
#          bias_importance=torch.tensor([1.0, 1.0]),
#          ref_sparsity=8 / 12.0,
#          nncf_config=MovementSparsityConfigBuilder().sparse_structure_by_scopes([["block", [2, 2], "{re}.*"]])),
#     dict(weight=torch.arange(-4, 4).float().reshape(4, 2),
#          bias=torch.ones([4]),
#          weight_importance=torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]]),
#          bias_importance=torch.tensor([1.0, 1.0, 0.0, 1.0]),
#          ref_sparsity=5 / 12.0,
#          nncf_config=MovementSparsityConfigBuilder().sparse_structure_by_scopes([["fine", [1, 1], "{re}.*"]]))
# ])
# def test_sparsity_statistics(config):
#     class BasicModel(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.linears = nn.Linear(2, 4, bias=True)

#         def forward(self, inputs):
#             return self.linears(inputs)

#     compression_ctrl, model = create_compressed_model(BasicModel(), config['nncf_config'].input_info([{"sample_size": [1, 2]}]).build())
#     for sparse_module in compression_ctrl.sparsified_module_info:
#           # HOW to set data? load a state_dict?
#         sparse_module.module.weight.data = config['weight']
#         sparse_module.module.bias.data = config['bias']
#         sparse_module.operand._weight_importance.data = config['weight_importance']
#         sparse_module.operand._bias_importance.data = config['bias_importance']
#         calculated_sparsity = compression_ctrl.statistics().movement_sparsity.model_statistics.sparsity_level_for_layers
#         print(calculated_sparsity)
#         assert calculated_sparsity == approx(config['ref_sparsity'])


if __name__ == "__main__":
    test_can_create_movement_sparsity_layers(MovementSparsityConfigBuilder())
    # test_can_run_full_pipeline(Path('/tmp'), MovementSparsityConfigBuilder)
    # test_importance_score_update(Path('/tmp'), MovementSparsityConfigBuilder())
    pass
