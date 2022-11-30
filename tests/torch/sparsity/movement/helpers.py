from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import Mock

from datasets import Dataset  # pylint: disable=no-name-in-module
import numpy as np
from pytest import approx
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
from transformers import AutoModelForAudioClassification
from transformers import AutoModelForImageClassification
from transformers import AutoModelForSequenceClassification
from transformers import BertConfig
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from transformers import SwinConfig
from transformers import TrainingArguments
from transformers import Wav2Vec2Config
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_callback import TrainerControl
from transformers.trainer_callback import TrainerState

from nncf import NNCFConfig
from nncf.api.compression import CompressionAlgorithmController
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.utils.tensorboard import prepare_for_tensorboard
from nncf.experimental.torch.sparsity.movement.algo import MovementSparsifier
from nncf.torch.nncf_network import NNCFNetwork


def mock_linear_nncf_node(in_features: int = 1, out_features: int = 1,
                          bias: bool = True, node_name: str = 'linear') -> NNCFNode:
    graph = NNCFGraph()
    linear = graph.add_nncf_node(
        node_name, 'linear', Mock(),
        LinearLayerAttributes(True, in_features, out_features, bias=bias))
    return linear


def ensure_tensor(data, dtype=torch.float, device=torch.device('cpu')) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype=dtype, device=device)
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype, device=device)
    return torch.tensor(data, dtype=dtype, device=device)


def initialize_sparsifer_parameters(operand: MovementSparsifier,
                                    linspace_start: float = -1, linspace_end: float = 1):
    with torch.no_grad():
        device = operand.weight_importance.device
        weight_init_tensor = torch.linspace(linspace_start, linspace_end,
                                            steps=operand.weight_importance.numel(),
                                            device=device)\
            .reshape_as(operand.weight_importance)
        operand.weight_importance.copy_(weight_init_tensor)
        if operand.prune_bias:
            bias_init_tensor = torch.linspace(linspace_start, linspace_end,
                                              steps=operand.bias_importance.numel(),
                                              device=device)\
                .reshape_as(operand.bias_importance)
            operand.bias_importance.copy_(bias_init_tensor)


def is_roughly_non_decreasing(x_list, atol: float = 0.01) -> bool:
    x_list = list(x_list)
    assert atol >= 0
    return all(a <= b + atol for a, b in zip(x_list[:-1], x_list[1:]))


def is_roughly_of_same_value(x_list, atol: float = 1e-6) -> bool:
    x_list = list(x_list)
    assert atol >= 0
    return all(x == approx(x_list[0], abs=atol) for x in x_list[1:])


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
    def __init__(self, sparse_structure_by_scopes: Optional[List[Dict]] = None,
                 ignored_scopes: Optional[List[str]] = None,
                 scheduler_params: Optional[SchedulerParams] = None,
                 **scheduler_overrides):
        self.scheduler_params = scheduler_params or SchedulerParams()
        for k, v in scheduler_overrides.items():
            assert hasattr(self.scheduler_params, k)
            setattr(self.scheduler_params, k, v)
        self.sparse_structure_by_scopes = sparse_structure_by_scopes or []
        self.ignored_scopes = ignored_scopes or []

    def to_dict(self):
        return {
            "algorithm": "movement_sparsity",
            "params": self.scheduler_params.__dict__,
            "sparse_structure_by_scopes": self.sparse_structure_by_scopes,
            "ignored_scopes": self.ignored_scopes,
        }


class TransformerBlockInfo:
    def __init__(self, num_hidden_layers: int = 1,
                 hidden_size: int = 4,
                 intermediate_size: int = 3,
                 dim_per_head: int = 2):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dim_per_head = dim_per_head


class TransformerBlockItemOrderedDict(OrderedDict):
    def __init__(self, mhsa_q: Any, mhsa_k: Any, mhsa_v: Any,
                 mhsa_o: Any, ffn_i: Any, ffn_o: Any) -> None:
        super().__init__(mhsa_q=mhsa_q, mhsa_k=mhsa_k, mhsa_v=mhsa_v,
                         mhsa_o=mhsa_o, ffn_i=ffn_i, ffn_o=ffn_o)


class BaseMockRunRecipe(ABC):
    model_family: str
    supports_structured_masking: bool
    default_model_config = PretrainedConfig()
    default_algo_config = NNCFAlgoConfig()

    def __init__(self, model_config: PretrainedConfig,
                 algo_config: NNCFAlgoConfig,
                 log_dir=None) -> None:
        self.model_config = model_config
        self.algo_config = algo_config
        self.model_keys = set(self.model_config.__dict__)
        self.scheduler_keys = set(self.algo_config.scheduler_params.__dict__)
        self.algo_keys = set(self.algo_config.__dict__)
        self.set_log_dir(log_dir)

    @classmethod
    def from_default(cls, log_dir=None, **override_kwargs):
        model_config = deepcopy(cls.default_model_config)
        algo_config = deepcopy(cls.default_algo_config)
        scheduler_config = algo_config.scheduler_params
        model_keys = set(model_config.__dict__)
        scheduler_keys = set(scheduler_config.__dict__)
        algo_keys = set(algo_config.__dict__)
        for key, value in override_kwargs.items():
            if key in model_keys:
                setattr(model_config, key, value)
            elif key in algo_keys:
                setattr(algo_config, key, value)
            elif key in scheduler_keys:
                setattr(scheduler_config, key, value)
            else:
                raise ValueError(f'Unknown config: {key}')
        return cls(model_config, algo_config, log_dir)

    @property
    @abstractmethod
    def model_input_info(self) -> List[dict]:
        pass

    @property
    @abstractmethod
    def transformer_block_info(self) -> List[TransformerBlockInfo]:
        pass

    @property
    def model(self) -> torch.nn.Module:
        torch_model = self._create_model()
        g = torch.Generator()
        g.manual_seed(42)
        with torch.no_grad():
            for _, parameter in torch_model.named_parameters():
                parameter.normal_(generator=g)
        return torch_model

    @property
    def nncf_config(self) -> NNCFConfig:
        config_dict = {
            "input_info": self.model_input_info,
            "compression": self.algo_config.to_dict()}
        if self.log_dir is not None:
            config_dict['log_dir'] = str(self.log_dir)
        return NNCFConfig.from_dict(config_dict)

    @property
    def scheduler_params(self):
        return self.algo_config.scheduler_params

    @staticmethod
    @abstractmethod
    def get_nncf_modules_in_transformer_block_order(
            compressed_model: NNCFNetwork) -> List[TransformerBlockItemOrderedDict]:
        pass

    @abstractmethod
    def _create_model(self) -> torch.nn.Module:
        pass

    def set_log_dir(self, log_dir=None):
        self.log_dir = str(log_dir)
        if log_dir is not None:
            Path(log_dir).mkdir(exist_ok=True, parents=True)

    def get(self, key: str):
        if key == 'log_dir':
            return self.log_dir
        if key in self.model_keys:
            return self.model_config.__dict__[key]
        if key in self.algo_keys:
            return self.algo_config.__dict__[key]
        if key in self.scheduler_keys:
            return self.algo_config.scheduler_params.__dict__[key]
        raise KeyError(f'{key} not found.')

    def set(self, key: str, value):
        if key == 'log_dir':
            self.set_log_dir(value)
        elif key in self.model_keys:
            setattr(self.model_config, key, value)
        elif key in self.algo_keys:
            setattr(self.algo_config, key, value)
        elif key in self.scheduler_keys:
            setattr(self.algo_config.scheduler_params, key, value)
        else:
            raise KeyError(f'"{key}" not found.')

    def generate_mock_dataset(self, num_samples: int = 16, seed: int = 42) -> Dataset:
        g = torch.Generator()
        g.manual_seed(seed)
        input_dict = {}
        for input_info in self.model_input_info:
            shape = list(input_info.get('sample_size'))
            keyword = input_info.get('keyword')
            if input_info.get('type', 'float') == 'float':
                tensor = torch.randn((num_samples, *shape[1:]), dtype=torch.float, generator=g)
            else:
                tensor = torch.randint(0, 2, (num_samples, *shape[1:]), generator=g)
            input_dict[keyword] = tensor
        input_dict['labels'] = torch.arange(self.model_config.num_labels).repeat(
            num_samples // self.model_config.num_labels + 1)[:num_samples]
        return Dataset.from_dict(input_dict)


class Wav2Vec2RunRecipe(BaseMockRunRecipe):
    model_family = 'huggingface_wav2vec2'
    supports_structured_masking = True
    default_model_config = Wav2Vec2Config(
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=3,
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
        sparse_structure_by_scopes=[
            {"mode": "block", "sparse_factors": [2, 2], "target_scopes": "{re}Wav2Vec2Attention"},
            {"mode": "per_dim", "axis": 0, "target_scopes": "{re}intermediate_dense"},
            {"mode": "per_dim", "axis": 1, "target_scopes": "{re}output_dense"},
        ],
        ignored_scopes=["{re}feature_extractor"],
        scheduler_params=SchedulerParams(),
    )

    def _create_model(self) -> torch.nn.Module:
        return AutoModelForAudioClassification.from_config(self.model_config)

    @property
    def model_input_info(self) -> List[dict]:
        return [{"sample_size": [1, 32], "keyword": "input_values"}]

    @property
    def transformer_block_info(self) -> List[TransformerBlockInfo]:
        model_config = self.model_config
        return [TransformerBlockInfo(
            num_hidden_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            dim_per_head=model_config.hidden_size // model_config.num_attention_heads
        )]

    @staticmethod
    def get_nncf_modules_in_transformer_block_order(
            compressed_model: NNCFNetwork) -> List[TransformerBlockItemOrderedDict]:
        modules = []
        for block in compressed_model.nncf_module.wav2vec2.encoder.layers:
            modules.append(TransformerBlockItemOrderedDict(
                block.attention.q_proj,
                block.attention.k_proj,
                block.attention.v_proj,
                block.attention.out_proj,
                block.feed_forward.intermediate_dense,
                block.feed_forward.output_dense)
            )
        return modules


class BertRunRecipe(BaseMockRunRecipe):
    model_family = 'huggingface_bert'
    supports_structured_masking = True
    default_model_config = BertConfig(
        hidden_size=4,
        intermediate_size=3,
        max_position_embeddings=128,
        num_attention_heads=2,
        num_hidden_layers=1,
        vocab_size=10,
        num_labels=2,
        mhsa_qkv_bias=True,
        mhsa_o_bias=True,
        ffn_bias=True
    )
    default_algo_config = NNCFAlgoConfig(
        sparse_structure_by_scopes=[
            {"mode": "block", "sparse_factors": [2, 2], "target_scopes": "{re}attention"},
            {"mode": "per_dim", "axis": 0, "target_scopes": "{re}BertIntermediate"},
            {"mode": "per_dim", "axis": 1, "target_scopes": "{re}BertOutput"},
        ],
        ignored_scopes=["{re}embedding", "{re}pooler", "{re}classifier"],
        scheduler_params=SchedulerParams(),
    )

    def __init__(self, model_config: BertConfig,
                 algo_config: NNCFAlgoConfig,
                 log_dir=None) -> None:
        super().__init__(model_config, algo_config, log_dir)
        extra_model_keys = {'mhsa_qkv_bias', 'mhsa_o_bias', 'ffn_bias'}
        self.model_keys = self.model_keys.union(extra_model_keys)
        for key in extra_model_keys:
            value = getattr(self.model_config, key, True)
            setattr(self.model_config, key, value)

    def _create_model(self) -> torch.nn.Module:
        model = AutoModelForSequenceClassification.from_config(self.model_config)
        for block in model.bert.encoder.layer:
            if not self.model_config.mhsa_qkv_bias:
                block.attention.self.query.bias = None
                block.attention.self.key.bias = None
                block.attention.self.value.bias = None
            if not self.model_config.mhsa_o_bias:
                block.attention.output.dense.bias = None
            if not self.model_config.ffn_bias:
                block.intermediate.dense.bias = None
                block.output.dense.bias = None
        return model

    @property
    def model_input_info(self) -> List[dict]:
        dim = self.model_config.max_position_embeddings
        return [
            {"sample_size": [1, dim], "type": "long", "keyword": "input_ids"},
            {"sample_size": [1, dim], "type": "long", "keyword": "token_type_ids"},
            {"sample_size": [1, dim], "type": "long", "keyword": "position_ids"},
            {"sample_size": [1, dim], "type": "long", "keyword": "attention_mask"},
        ]

    @property
    def transformer_block_info(self) -> List[TransformerBlockInfo]:
        model_config = self.model_config
        return [TransformerBlockInfo(
            num_hidden_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            dim_per_head=model_config.hidden_size // model_config.num_attention_heads
        )]

    @staticmethod
    def get_nncf_modules_in_transformer_block_order(
            compressed_model: NNCFNetwork) -> List[TransformerBlockItemOrderedDict]:
        modules = []
        for block in compressed_model.nncf_module.bert.encoder.layer:
            modules.append(TransformerBlockItemOrderedDict(
                block.attention.self.query,
                block.attention.self.key,
                block.attention.self.value,
                block.attention.output.dense,
                block.intermediate.dense,
                block.output.dense)
            )
        return modules


class SwinRunRecipe(BaseMockRunRecipe):
    model_family = 'huggingface_swin'
    supports_structured_masking = True
    default_model_config = SwinConfig(
        image_size=4,
        patch_size=1,
        num_channels=3,
        embed_dim=4,
        depths=[1],
        num_heads=[2],
        window_size=2,
        mlp_ratio=3 / 4,
        num_labels=2,
        qkv_bias=True,
    )
    default_algo_config = NNCFAlgoConfig(
        sparse_structure_by_scopes=[
            {"mode": "block", "sparse_factors": [2, 2], "target_scopes": "{re}attention"},
            {"mode": "per_dim", "axis": 0, "target_scopes": "{re}SwinIntermediate"},
            {"mode": "per_dim", "axis": 1, "target_scopes": "{re}SwinOutput"},
        ],
        ignored_scopes=["{re}embedding", "{re}pooler", "{re}classifier"],
        scheduler_params=SchedulerParams(),
    )

    def _create_model(self) -> torch.nn.Module:
        return AutoModelForImageClassification.from_config(self.model_config)

    @property
    def model_input_info(self) -> List[dict]:
        img_size = self.model_config.image_size
        return [{"sample_size": [1, self.model_config.num_channels, img_size, img_size],
                 "keyword": 'pixel_values'}]

    @property
    def transformer_block_info(self) -> List[TransformerBlockInfo]:
        model_config = self.model_config
        info_list = []
        for i, (depth, num_head) in enumerate(zip(model_config.depths, model_config.num_heads)):
            hidden_size = model_config.embed_dim * int(2 ** i)
            intermediate_size = int(hidden_size * model_config.mlp_ratio)
            dim_per_head = hidden_size // num_head
            info_list.append(TransformerBlockInfo(
                num_hidden_layers=depth,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dim_per_head=dim_per_head
            ))
        return info_list

    @staticmethod
    def get_nncf_modules_in_transformer_block_order(
            compressed_model: NNCFNetwork) -> List[TransformerBlockItemOrderedDict]:
        modules = []
        for layer in compressed_model.nncf_module.swin.encoder.layers:
            for block in layer.blocks:
                modules.append(TransformerBlockItemOrderedDict(
                    block.attention.self.query,
                    block.attention.self.key,
                    block.attention.self.value,
                    block.attention.output.dense,
                    block.intermediate.dense,
                    block.output.dense)
                )
        return modules


class LinearForClassification(PreTrainedModel):

    def __init__(self, input_size: int = 4, bias: bool = True, num_classes: int = 2):
        super().__init__(PretrainedConfig())
        self.model = torch.nn.Linear(input_size, num_classes, bias=bias)

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


class Conv2dForClassification(LinearForClassification):

    def __init__(self, input_size: int = 4, bias: bool = True, num_classes: int = 2):
        super().__init__(input_size, bias, num_classes)
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, num_classes, kernel_size=3, stride=1, padding=1, bias=bias),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )


class Conv2dPlusLinearForClassification(LinearForClassification):

    def __init__(self, input_size: int = 4, bias: bool = True, num_classes: int = 2):
        super().__init__(input_size, bias, num_classes)
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, num_classes, kernel_size=3, stride=1, padding=1, bias=bias),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(num_classes, num_classes, bias=bias)
        )


class LinearRunRecipe(BaseMockRunRecipe):
    model_family = 'linear'
    supports_structured_masking = False
    default_model_config = PretrainedConfig(
        num_classes=2,
        input_size=4,
        bias=True
    )
    default_algo_config = NNCFAlgoConfig(
        enable_structured_masking=False
    )

    def _create_model(self) -> torch.nn.Module:
        model_config = self.model_config
        return LinearForClassification(input_size=model_config.input_size,
                                       bias=model_config.bias,
                                       num_classes=model_config.num_classes)

    @property
    def model_input_info(self) -> List[dict]:
        return [{"sample_size": [1, self.model_config.input_size], "keyword": "tensor"}]

    @property
    def transformer_block_info(self) -> List[TransformerBlockInfo]:
        return []

    @staticmethod
    def get_nncf_modules_in_transformer_block_order(
            compressed_model: NNCFNetwork) -> List[TransformerBlockItemOrderedDict]:
        return []


class Conv2dRunRecipe(LinearRunRecipe):
    model_family = 'conv2d'
    supports_structured_masking = False
    default_model_config = PretrainedConfig(
        num_classes=2,
        input_size=4,
        bias=True
    )
    default_algo_config = NNCFAlgoConfig(
        enable_structured_masking=False
    )

    def _create_model(self) -> torch.nn.Module:
        model_config = self.model_config
        return Conv2dForClassification(input_size=model_config.input_size,
                                       bias=model_config.bias,
                                       num_classes=model_config.num_classes)

    @property
    def model_input_info(self) -> List[dict]:
        return [{"sample_size": [1, 3, self.model_config.input_size, self.model_config.input_size],
                 "keyword": "tensor"}]


class Conv2dPlusLinearRunrecipe(LinearRunRecipe):
    model_family = 'conv2d+linear'
    supports_structured_masking = False
    default_model_config = PretrainedConfig(
        num_classes=2,
        input_size=4,
        bias=True
    )
    default_algo_config = NNCFAlgoConfig(
        enable_structured_masking=False
    )

    def _create_model(self) -> torch.nn.Module:
        model_config = self.model_config
        return Conv2dPlusLinearForClassification(input_size=model_config.input_size,
                                                 bias=model_config.bias,
                                                 num_classes=model_config.num_classes)

    @property
    def model_input_info(self) -> List[dict]:
        return [{"sample_size": [1, 3, self.model_config.input_size, self.model_config.input_size],
                 "keyword": "tensor"}]


class CompressionTrainer(Trainer):
    def __init__(self,
                 compression_ctrl: Optional[CompressionAlgorithmController],
                 *args,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 **kwargs):
        self.compression_ctrl = compression_ctrl
        if compression_ctrl is not None:
            if not callbacks:
                compression_callback = CompressionCallback(compression_ctrl)
                callbacks = [compression_callback]
                self.compression_callback = compression_callback
            else:
                assert len(callbacks) == 1
                self.compression_callback = callbacks[0]
        super().__init__(callbacks=callbacks, *args, **kwargs)
        if not (self.args.local_rank == -1 or self.args.no_cuda or compression_ctrl is None):
            compression_ctrl.distributed()

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if self.compression_ctrl is not None:
            loss_compress = self.compression_ctrl.loss()
            loss = loss + loss_compress
        return (loss, outputs) if return_outputs else loss


class CompressionCallback(TrainerCallback):
    def __init__(self, compression_ctrl: CompressionAlgorithmController) -> None:
        self.compression_ctrl = compression_ctrl
        self._compression_log_by_step = OrderedDict()

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.compression_ctrl.scheduler.epoch_step()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.compression_ctrl.scheduler.step()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        stats = self.compression_ctrl.statistics()
        stat_dict = prepare_for_tensorboard(stats)
        stat_dict.update(step=state.global_step, epoch=state.epoch)
        self._compression_log_by_step[state.global_step] = stat_dict

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._training_log = state.log_history

    def get_compression_log(self, step_starts_from_1=True):
        if step_starts_from_1:
            return self._compression_log_by_step
        return {(step - 1): log for step, log in self._compression_log_by_step.items()}

    def get_train_log(self):
        return self._training_log


def build_compression_trainer(output_dir,
                              compression_ctrl: CompressionAlgorithmController,
                              compressed_model: NNCFNetwork,
                              train_dataset: Dataset,
                              eval_dataset: Optional[Dataset] = None,
                              callback: Optional[CompressionCallback] = None,
                              batch_size: int = 1,
                              **training_kwargs) -> CompressionTrainer:
    training_args = dict(
        output_dir=Path(output_dir),
        label_names=["labels"],
        evaluation_strategy="epoch",
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=6,
        learning_rate=1e-3,
        optim="adamw_torch",
        remove_unused_columns=False,
        seed=42,
        full_determinism=True,
        report_to="none",
        disable_tqdm=True,
        no_cuda=True,
    )
    if eval_dataset is None:
        training_args['evaluation_strategy'] = 'no'
    training_args.update(training_kwargs)
    training_args = TrainingArguments(**training_args)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return dict(acc=(predictions == labels).mean())

    if callback is None:
        callback = CompressionCallback(compression_ctrl)

    trainer = CompressionTrainer(
        model=compressed_model,
        args=training_args,
        compression_ctrl=compression_ctrl,
        callbacks=[callback],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    return trainer
