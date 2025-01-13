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
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datasets
import evaluate
import jstyleson
import numpy as np
from transformers.training_args import ParallelMode

# isort: off
from nncf import NNCFConfig
from nncf.api.compression import CompressionAlgorithmController
from nncf.common.utils.tensorboard import prepare_for_tensorboard
from nncf.torch import create_compressed_model

from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import EvalPrediction
from transformers import HfArgumentParser
from transformers import set_seed
from transformers.trainer import Trainer
from transformers.trainer import TrainerCallback
from transformers.trainer import TrainerControl
from transformers.trainer import TrainerState
from transformers.trainer import TrainingArguments


quick_check_num = 10
task_to_sample_keys = {
    "mrpc": ("sentence1", "sentence2"),
    "sst2": ("sentence",),
}
dataset_columns = ["labels", "input_ids", "token_type_ids", "attention_mask", "position_ids"]


def parse_args() -> Tuple[argparse.Namespace, TrainingArguments]:
    parser = argparse.ArgumentParser("GLUE")
    parser.add_argument(
        "--task_name",
        type=str,
        default="mrpc",
        help=f"Task name for GLUE. Supported tasks: {list(task_to_sample_keys)}.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum length for model input sequences.")
    parser.add_argument("--nncf_config", type=str, default=None, help="Path to NNCF configuration json file.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether to disable cuda devices.")
    parser.add_argument("--seed", type=int, default=42, help="Experiment seed for training and datasets.")
    parser.add_argument(
        "--quick_check",
        action="store_true",
        help="If set True, will train the model without pretrained weights on only " f"{quick_check_num} samples.",
    )

    args, other_args = parser.parse_known_args()
    retval = HfArgumentParser(TrainingArguments).parse_args_into_dataclasses(other_args)
    training_args: TrainingArguments = retval[0]

    # post parser checks and overrides
    assert args.task_name in task_to_sample_keys, f"Task name should be in {list(task_to_sample_keys)}."
    training_args.no_cuda = args.no_cuda
    training_args.data_seed = args.seed
    training_args.seed = args.seed
    training_args.label_names = ["labels"]
    training_args.remove_unused_columns = False
    training_args.overwrite_output_dir = True
    training_args.report_to = []
    return args, training_args


class CompressionCallback(TrainerCallback):
    def __init__(self, compression_ctrl: CompressionAlgorithmController):
        self.compression_ctrl = compression_ctrl
        self.compression_logs = []

    def on_epoch_begin(self, *args, **kwargs):
        self.compression_ctrl.scheduler.epoch_step()

    def on_step_begin(self, *args, **kwargs):
        self.compression_ctrl.scheduler.step()

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        stats = self._gather_compression_stats(state.global_step, state.epoch)
        self.compression_logs.append(stats)

    def _gather_compression_stats(self, step: int, epoch: Optional[float]) -> Dict[str, float]:
        status = {"step": step}
        if epoch is not None:
            status["epoch"] = round(epoch, 2)
        stats = prepare_for_tensorboard(self.compression_ctrl.statistics())
        result = {**status, **stats}
        return result


class CompressionTrainer(Trainer):
    def __init__(
        self,
        compression_ctrl: Optional[CompressionAlgorithmController],
        *args,
        callbacks: Optional[List[TrainerCallback]] = None,
        **kwargs,
    ):
        self.compression_ctrl = compression_ctrl
        self._compression_callback = None
        if compression_ctrl is not None:
            self._compression_callback = CompressionCallback(compression_ctrl)
            callbacks = [self._compression_callback] + (callbacks or [])
        super().__init__(callbacks=callbacks, *args, **kwargs)
        if (
            self.args.parallel_mode == ParallelMode.DISTRIBUTED
            and not self.args.no_cuda
            and compression_ctrl is not None
        ):
            compression_ctrl.distributed()

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if self.compression_ctrl is not None:
            loss_compress = self.compression_ctrl.loss()
            loss = loss + loss_compress
        return (loss, outputs) if return_outputs else loss

    def get_compression_logs(self) -> Optional[List[Dict[str, float]]]:
        if self._compression_callback is None:
            return None
        return self._compression_callback.compression_logs


def prepare_dataset(args, training_args):
    raw_datasets = datasets.load_dataset("glue", args.task_name)
    num_labels = len(raw_datasets["train"].features["label"].names)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def tokenize_fn(samples):
        sample_keys = task_to_sample_keys[args.task_name]
        max_length = min(args.max_seq_length, tokenizer.model_max_length)
        result = tokenizer(
            *(samples[key] for key in sample_keys), padding="max_length", max_length=max_length, truncation=True
        )
        result["position_ids"] = list(range(max_length))
        return result

    def process_dataset(dataset):
        if args.quick_check:
            dataset = dataset.select(range(quick_check_num))
        dataset = dataset.map(tokenize_fn)
        dataset = dataset.rename_column("label", "labels")
        columns_to_remove = set(dataset.column_names) - set(dataset_columns)
        dataset = dataset.remove_columns(list(columns_to_remove))
        return dataset

    train_dataset = eval_dataset = None
    with training_args.main_process_first():
        if training_args.do_train:
            train_dataset = process_dataset(raw_datasets["train"])
        if training_args.do_eval:
            eval_dataset = process_dataset(raw_datasets["validation"])
    return train_dataset, eval_dataset, num_labels


def prepare_model(args: argparse.Namespace, training_args: TrainingArguments, num_labels: int):
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    if args.quick_check:
        model = AutoModelForSequenceClassification.from_config(config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    return model


def main():
    args, training_args = parse_args()
    if args.quick_check:
        print(
            "This run is for quick check. We will train the model without pretrained "
            f"weights on {quick_check_num} training samples only."
        )
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    if training_args.seed is not None:
        set_seed(training_args.seed)

    train_dataset, eval_dataset, num_labels = prepare_dataset(args, training_args)
    model = prepare_model(args, training_args, num_labels)

    # wrap with nncf if specified
    compression_ctrl = None
    if args.nncf_config is not None:
        nncf_config = NNCFConfig.from_json(args.nncf_config)
        if nncf_config.get("log_dir", None) is None:
            nncf_config["log_dir"] = training_args.output_dir
        Path(nncf_config["log_dir"]).mkdir(parents=True, exist_ok=True)
        compression_ctrl, model = create_compressed_model(model, nncf_config)

    # trainer
    metric = evaluate.load("glue", args.task_name)

    def compute_metrics(p: EvalPrediction):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(logits, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        return result

    trainer = CompressionTrainer(
        compression_ctrl=compression_ctrl,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # do training & evaluation
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        compression_logs = trainer.get_compression_logs()
        if compression_logs:
            trainer.log_metrics("compression", compression_logs[-1])
            trainer.save_metrics("compression", compression_logs[-1])
            compression_log_str = jstyleson.dumps(compression_logs, indent=2) + "\n"
            compression_log_path = Path(training_args.output_dir, "compression_state.json")
            with open(compression_log_path, "w", encoding="utf-8") as f:
                f.write(compression_log_str)
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
