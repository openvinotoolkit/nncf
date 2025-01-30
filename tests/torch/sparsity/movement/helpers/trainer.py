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
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset  # pylint: disable=no-name-in-module
from transformers import TrainingArguments
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_callback import TrainerControl
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from nncf.api.compression import CompressionAlgorithmController
from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.common.utils.tensorboard import prepare_for_tensorboard
from nncf.torch.nncf_network import NNCFNetwork

CTRL_STATE_NAME = BaseCompressionAlgorithmController.CONTROLLER_STATE
NNCF_COMPRESSION_STATE_NAME = "nncf_compression_state.pt"


class CompressionTrainer(Trainer):
    def __init__(
        self,
        compression_ctrl: Optional[CompressionAlgorithmController],
        *args,
        callbacks: Optional[List[TrainerCallback]] = None,
        **kwargs,
    ):
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

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        state_path = Path(str(resume_from_checkpoint), NNCF_COMPRESSION_STATE_NAME)
        if self.compression_ctrl is not None and state_path.is_file():
            compression_state = torch.load(state_path)
            self.compression_ctrl.load_state(compression_state.get(CTRL_STATE_NAME, {}))
        len_dataloader = len(self.get_train_dataloader())
        steps_per_epoch = max(1, len_dataloader // self.args.gradient_accumulation_steps)
        if (self.compression_ctrl.scheduler.current_step + 1) % steps_per_epoch != 0:
            # The beginning of an incomplete epoch when resume
            self.compression_callback.should_skip_next_epoch_step()


class CompressionCallback(TrainerCallback):
    def __init__(self, compression_ctrl: CompressionAlgorithmController) -> None:
        self.compression_ctrl = compression_ctrl
        self._compression_log_by_step = OrderedDict()
        self._skip_next_epoch_step = False

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self._skip_next_epoch_step:
            self.compression_ctrl.scheduler.epoch_step()
        self._skip_next_epoch_step = False

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.compression_ctrl.scheduler.step()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        stats = self.compression_ctrl.statistics()
        stat_dict = prepare_for_tensorboard(stats)
        stat_dict.update(step=state.global_step, epoch=state.epoch)
        self._compression_log_by_step[state.global_step] = stat_dict

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        compression_state = self.compression_ctrl.get_compression_state()
        save_path = Path(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}", NNCF_COMPRESSION_STATE_NAME)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(compression_state, save_path)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._training_log = state.log_history

    def get_compression_log(self, step_starts_from_1=True):
        if step_starts_from_1:
            return self._compression_log_by_step
        return {(step - 1): log for step, log in self._compression_log_by_step.items()}

    def get_train_log(self):
        return self._training_log

    def should_skip_next_epoch_step(self):
        self._skip_next_epoch_step = True


def build_compression_trainer(
    output_dir,
    compression_ctrl: CompressionAlgorithmController,
    compressed_model: NNCFNetwork,
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Optional[Dataset] = None,
    callback: Optional[CompressionCallback] = None,
    batch_size: int = 1,
    num_train_epochs: int = 6,
    **training_kwargs,
) -> CompressionTrainer:
    evaluation_strategy = "no" if eval_dataset is None else "epoch"
    training_args = dict(
        output_dir=Path(output_dir),
        label_names=["labels"],
        evaluation_strategy=evaluation_strategy,
        save_strategy="steps",
        logging_strategy="steps",
        save_steps=500,
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=1e-3,
        optim="adamw_torch",
        remove_unused_columns=False,
        seed=42,
        data_seed=42,
        full_determinism=True,
        report_to="none",
        disable_tqdm=True,
        no_cuda=True,
    )
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
