from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
import pytest
import torch
import torch.nn as nn
from datasets import load_dataset
from nncf import NNCFConfig
from nncf.api.compression import CompressionAlgorithmController
from nncf.experimental.torch.sparsity.movement.algo import MovementSparsifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer as BaseTrainer
from transformers import TrainingArguments, BertConfig
from transformers import Wav2Vec2Config
from transformers import AutoModelForAudioClassification
from transformers import SwinConfig
from transformers import AutoModelForImageClassification
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


def bert_tiny_unpretrained():
    model_cfg = {
        "hidden_size": 4,
        "intermediate_size": 3,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "vocab_size": 30522,
    }
    return AutoModelForSequenceClassification.from_config(BertConfig(**model_cfg, num_labels=5))


def wav2vec2_model():
    config = Wav2Vec2Config(
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=6,
        conv_dim=(4, 4),
        conv_stride=(1, 1),
        conv_kernel=(3, 3),
        num_conv_pos_embeddings=3,
        num_conv_pos_embedding_groups=1,
        proj_codevector_dim=4,
        classifier_proj_size=3
    )
    return AutoModelForAudioClassification.from_config(config)


def swin_model():
    config = SwinConfig()  # TODO(yujie): change config for a smaller model
    return AutoModelForImageClassification.from_config(config)


class BaseMockRunRecipe:
    model_family: str

    def __init__(self, model_config, algo_config) -> None:
        self.model_config = model_config
        self.algo_config = algo_config

    @property
    def model(self):
        pass

    @property
    def nncf_config(self):
        pass

    @property
    def mock_dataset(self):
        pass

    @staticmethod
    def get_nncf_modules_in_transformer_block_order(compressed_model):
        """Returns the NNCF modules in usual transformer block order, 
        i.e., List[Tuple(query, key, value, output, feedforward_in, feedforward_out)]
        """
        return []


class SchedulerParams:
    def __init__(self, power: int = 3,
                 warmup_start_epoch: int = 1,
                 warmup_end_epoch: int = 3,
                 init_importance_threshold: float = -1.0,
                 final_importance_threshold: float = 0.0,
                 importance_regularization_factor: float = 0.1,
                 steps_per_epoch: Optional[int] = 4,
                 enable_structured_masking: bool = True):
        self.power = power
        self.warmup_start_epoch = warmup_start_epoch
        self.warmup_end_epoch = warmup_end_epoch
        self.init_importance_threshold = init_importance_threshold
        self.final_importance_threshold = final_importance_threshold
        self.importance_regularization_factor = importance_regularization_factor
        self.steps_per_epoch = steps_per_epoch
        self.enable_structured_masking = enable_structured_masking


class NNCFAlgoConfig:
    def __init__(self, sparse_structure_by_scopes=[], ignored_scopes=[],
                 scheduler_params=None, **scheduler_overrides):
        self.scheduler_params = scheduler_params or SchedulerParams()
        for k, v in scheduler_overrides.items():
            assert hasattr(self.scheduler_params, k)
            setattr(self.scheduler_params, k, v)
        self.sparse_structure_by_scopes = sparse_structure_by_scopes
        self.ignored_scopes = ignored_scopes

    def to_dict(self):
        return {
            "algorithm": "movement_sparsity",
            "params": self.scheduler_params.__dict__,
            "sparse_structure_by_scopes": self.sparse_structure_by_scopes,
            "ignored_scopes": self.ignored_scopes,
        }


class Wav2Vec2RunRecipe(BaseMockRunRecipe):
    model_family = 'huggingface_wav2vec2'
    default_model_config = Wav2Vec2Config(
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=6,
        conv_dim=(4, 4),
        conv_stride=(1, 1),
        conv_kernel=(3, 3),
        num_conv_pos_embeddings=3,
        num_conv_pos_embedding_groups=1,
        proj_codevector_dim=4,
        classifier_proj_size=3,
        num_labels=2,
    )

    default_algo_config = NNCFAlgoConfig(
        sparse_structure_by_scopes=[],
        ignored_scopes=[],
        scheduler_params=SchedulerParams(),
    )

    def __init__(self,
                 model_config: Optional[Wav2Vec2Config] = None,
                 algo_config: Optional[NNCFAlgoConfig] = None):
        model_config = model_config or Wav2Vec2RunRecipe.default_model_config
        algo_config = algo_config or Wav2Vec2RunRecipe.default_algo_config
        super().__init__(model_config, algo_config)

    @property
    def model(self):
        return AutoModelForAudioClassification.from_config(self.model_config)

    @staticmethod
    def get_nncf_modules_in_transformer_block_order(compressed_wav2vec2_model):
        modules = []
        for block in compressed_wav2vec2_model.nncf_module.wav2vec2.encoder.layers:
            modules.append(
                (block.attention.q_proj,
                 block.attention.k_proj,
                 block.attention.v_proj,
                 block.attention.out_proj,
                 block.feed_forward.intermediate_dense,
                 block.feed_forward.output_dense)
            )
        return modules

    @property
    def nncf_config(self):
        config_dict = {
            "input_info": [
                {"sample_size": [1, 32], "keyword": "input_values"}
            ],
            "compression": self.algo_config.to_dict()
        }
        return NNCFConfig.from_dict(config_dict)

    @property
    def mock_dataset(self):
        pass


class BertRunRecipe(BaseMockRunRecipe):
    model_family = 'huggingface_bert'
    default_model_config = BertConfig(
        hidden_size=4,
        intermediate_size=3,
        max_position_embeddings=512,
        num_attention_heads=2,
        num_hidden_layers=1,
        vocab_size=30522,
        num_labels=2,
    )
    default_algo_config = NNCFAlgoConfig(
        sparse_structure_by_scopes=[
            {"mode": "block", "sparse_factors": [2, 2], "target_scopes": "{re}.*attention*"},
            {"mode": "per_dim", "axis": 0, "target_scopes": "{re}.*BertIntermediate.*"},
            {"mode": "per_dim", "axis": 1, "target_scopes": "{re}.*BertOutput.*"},
        ],
        ignored_scopes=["{re}embedding", "{re}pooler", "{re}classifier"],
        scheduler_params=SchedulerParams(),
    )

    def __init__(self,
                 model_config: Optional[BertConfig] = None,
                 algo_config: Optional[NNCFAlgoConfig] = None):
        model_config = model_config or BertRunRecipe.default_model_config
        algo_config = algo_config or BertRunRecipe.default_algo_config
        super().__init__(model_config, algo_config)

    @property
    def model(self):
        return AutoModelForSequenceClassification.from_config(self.model_config)

    @property
    def nncf_config(self):
        return NNCFConfig.from_dict({
            "input_info": [
                {"sample_size": [1, 256], "type": "long", "keyword": "input_ids"},
                {"sample_size": [1, 256], "type": "long", "keyword": "token_type_ids"},
                {"sample_size": [1, 256], "type": "long", "keyword": "position_ids"},
                {"sample_size": [1, 256], "type": "long", "keyword": "attention_mask"},
            ],
            "compression": self.algo_config.to_dict()})

    @staticmethod
    def get_nncf_modules_in_transformer_block_order(compressed_bert_model):
        modules = []
        for block in compressed_bert_model.nncf_module.bert.encoder.layer:
            modules.append(
                (block.attention.self.query,
                 block.attention.self.key,
                 block.attention.self.value,
                 block.attention.output.dense,
                 block.intermediate.dense,
                 block.output.dense)
            )
        return modules


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
            "enable_structured_masking": True,
            "sparse_structure_by_scopes": [
                {"mode": "block", "sparse_factors": [16, 16], "target_scopes": "{re}.*attention*"},
                {"mode": "per_dim", "axis": 0, "target_scopes": "{re}.*BertIntermediate.*"},
                {"mode": "per_dim", "axis": 1, "target_scopes": "{re}.*BertOutput.*"},
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
                "params": dict(**args),
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
        num_train_epochs=6,
        learning_rate=1e-3,
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="none",
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


def initialize_sparsifer_parameters(operand: MovementSparsifier, mean: float = 0., std: float = 3.):
    torch.nn.init.normal_(operand.weight_importance, mean, std)
    if operand.prune_bias:
        torch.nn.init.normal_(operand.bias_importance, mean, std)
