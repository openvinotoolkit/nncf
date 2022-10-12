from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
import pytest
from datasets import load_dataset
from nncf import NNCFConfig
from nncf.api.compression import CompressionAlgorithmController
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer as BaseTrainer
from transformers import TrainingArguments
from transformers.trainer_callback import (TrainerCallback, TrainerControl,
                                           TrainerState)

MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
DATASET_NAME = "yelp_review_full"


# @pytest.fixture
def yelp_dataset():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset(DATASET_NAME)
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
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)


class ConfigBuilder:
    def __init__(self, **overrides):
        self._current_args = {
            "power": 3,
            "warmup_start_epoch": 1,
            "warmup_end_epoch": 3,
            "init_importance_threshold": -0.1,
            "final_importance_threshold": 0.0,
            "importance_regularization_factor": 0.2,
            "steps_per_epoch": 128 // 32,
            "update_per_optimizer_step": True,
            "sparse_structure_by_scopes": [
                ["block", [16, 16], "{re}.*attention*"],
                ["per_dim", [0], "{re}.*BertIntermediate.*"],
                ["per_dim", [1], "{re}.*BertOutput.*"],
            ],
            "ignored_scopes": ["{re}embedding", "{re}pooler", "{re}classifier"],
        }
        assert len(set(overrides.keys()) - set(self._current_args.keys())) == 0
        self.update(**overrides)

    def build(self, **tmp_overrides):
        args = deepcopy(self._current_args)
        sparse_structure_by_scopes = args.pop('sparse_structure_by_scopes', [])
        ignored_scopes = args.pop('ignored_scopes', [])
        config_dict = {
            "input_info": [
                {"sample_size": [1, 256], "type": "long", "keyword": "input_ids"},
                {"sample_size": [1, 256], "type": "long", "keyword": "token_type_ids"},
                {"sample_size": [1, 256], "type": "long", "keyword": "position_ids"},
                {"sample_size": [1, 256], "type": "long", "keyword": "attention_mask"},
            ],
            "compression": {
                "algorithm": "movement_sparsity",
                "params": dict(schedule="threshold_polynomial_decay", **args),
                "sparse_structure_by_scopes": sparse_structure_by_scopes,
                "ignored_scopes": ignored_scopes,
            },
        }
        nncf_config = NNCFConfig.from_dict(config_dict)
        nncf_config.update(tmp_overrides)
        return nncf_config

    def update(self, **overrides):
        self._current_args.update(overrides)
        return self

    def get(self, key):
        return deepcopy(self._current_args.get(key, None))

    def __getitem__(self, key):
        return self.get(key)

    def __call__(self, **overrides):
        self.update(**overrides)
        return self


class Trainer(BaseTrainer):
    def __init__(self, compression_ctrl, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compression_ctrl = compression_ctrl

    def compute_loss(self, model, inputs, return_outputs=False):
        loss_main, outputs = super().compute_loss(model, inputs, return_outputs=True)
        loss_compress = self.compression_ctrl.loss()
        loss = loss_main + loss_compress
        return (loss, outputs) if return_outputs else loss


class BaseCallback(TrainerCallback):
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
            print('Fill stage')
            mvmt_ctrl.reset_independent_structured_mask()
            mvmt_ctrl.resolve_structured_mask()
            mvmt_ctrl.populate_structured_mask()


def run_movement_pipeline(tmp_path, compression_ctrl, compressed_model,
                          callbacks: Optional[List[BaseCallback]] = None, **kwargs):  # temporarily remove fixture for model & dataset
    train_dataset, eval_dataset = yelp_dataset()
    args = dict(
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
    args.update(kwargs)
    training_args = TrainingArguments(**args)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return dict(acc=(predictions == labels).mean())

    if callbacks is None:
        callbacks = [BaseCallback(compression_ctrl)]
    trainer = Trainer(
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
