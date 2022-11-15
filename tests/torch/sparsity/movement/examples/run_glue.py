import argparse
import logging
from typing import Optional, List
from pathlib import Path
from collections import OrderedDict
from itertools import chain
from pprint import pformat

import torch
import torch.cuda
import numpy as np
from nncf import NNCFConfig
from nncf.torch import create_compressed_model
from nncf.torch.utils import is_main_process
from nncf.api.compression import CompressionAlgorithmController
from nncf.common.utils.tensorboard import prepare_for_tensorboard
import jstyleson as json

from datasets import load_dataset
from datasets import DatasetDict
import evaluate
import transformers
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import EvalPrediction
from transformers import HfArgumentParser
from transformers import set_seed
from transformers.trainer import Trainer
from transformers.trainer import TrainingArguments
from transformers.trainer import TrainerCallback
from transformers.trainer import TrainerState
from transformers.trainer import TrainerControl

quick_check_num = 10
task_to_sample_keys = {
    "mrpc": ("sentence1", "sentence2"),
    "sst2": ("sentence",),
}
dataset_columns = ['labels', 'input_ids', 'token_type_ids', 'attention_mask', 'position_ids']
nncf_logger = logging.getLogger('nncf')


def parse_args():
    parser = argparse.ArgumentParser('GLUE')
    parser.add_argument('--task_name', type=str, default='mrpc', help=f'Task name for GLUE. Supported tasks: {list(task_to_sample_keys)}.')
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased',
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum length for model input sequences.')
    parser.add_argument('--nncf_config', type=str, default=None, help='Path to NNCF configuration json file.')
    parser.add_argument('--no_cuda', action='store_true', help='Whether to disable cuda devices.')
    parser.add_argument('--quick_check', action='store_true',
                        help=f'If set, we will train the model without pretrained weights on only {quick_check_num} samples.')

    args, other_args = parser.parse_known_args()
    training_args, = HfArgumentParser(TrainingArguments).parse_args_into_dataclasses(other_args)

    # post parser checks and overrides
    assert args.task_name in task_to_sample_keys.keys(), f'Task name should be in {list(task_to_sample_keys)}.'
    training_args.no_cuda = args.no_cuda
    training_args.label_names = ["labels"]
    training_args.remove_unused_columns = False
    training_args.overwrite_output_dir = True
    training_args.report_to = []
    training_args.per_device_eval_batch_size = training_args.per_device_train_batch_size
    return args, training_args


class CompressionCallback(TrainerCallback):
    def __init__(self, compression_ctrl: CompressionAlgorithmController):
        self.compression_ctrl = compression_ctrl
        self._last_log_step = None

    def on_epoch_begin(self, *args, **kwargs):
        self.compression_ctrl.scheduler.epoch_step()

    def on_step_begin(self, *args, **kwargs):
        self.compression_ctrl.scheduler.step()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        step = state.global_step
        if step == self._last_log_step:
            return
        status = {"step": step}
        if state.epoch is not None:
            status["epoch"] = round(state.epoch, 2)
        stats = prepare_for_tensorboard(self.compression_ctrl.statistics())
        result = {**status, **stats}
        state.log_history.append(result)
        self._last_log_step = step


class CompressionTrainer(Trainer):
    def __init__(self,
                 compression_ctrl: Optional[CompressionAlgorithmController],
                 callbacks: Optional[List[TrainerCallback]] = None,
                 *args, **kwargs):
        self.compression_ctrl = compression_ctrl
        if compression_ctrl is not None:
            if callbacks:
                nncf_logger.warning('You passed a customized callback list to `CompressionTrainer`. '
                                    'Please ensure `.epoch_step()` and `.step()` are called for '
                                    '`compression_ctrl.scheduler`, otherwise compression may be incorrect.')
            else:
                compression_callback = CompressionCallback(compression_ctrl)
                callbacks = [compression_callback]
        self.compression_callbacks = callbacks
        super().__init__(callbacks=callbacks, *args, **kwargs)
        if not (self.args.local_rank == -1 or self.args.no_cuda or compression_ctrl is None):
            compression_ctrl.distributed()

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if self.compression_ctrl is not None:
            loss_compress = self.compression_ctrl.loss()
            loss = loss + loss_compress
        return (loss, outputs) if return_outputs else loss


def prepare_dataset(args, training_args):
    raw_datasets = load_dataset("glue", args.task_name)
    num_labels = len(raw_datasets["train"].features["label"].names)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def tokenize_fn(samples):
        sample_keys = task_to_sample_keys[args.task_name]
        max_length = min(args.max_seq_length, tokenizer.model_max_length)
        result = tokenizer(*(samples[key] for key in sample_keys),
                           padding="max_length",
                           max_length=max_length,
                           truncation=True)
        result['position_ids'] = list(range(max_length))
        return result

    def process_dataset(dataset):
        if args.quick_check:
            dataset = dataset.select(range(quick_check_num))
        dataset = dataset.map(tokenize_fn)
        dataset = dataset.rename_column('label', 'labels')
        columns_to_remove = set(dataset.column_names) - set(dataset_columns)
        dataset = dataset.remove_columns(list(columns_to_remove))
        return dataset

    train_dataset = eval_dataset = None
    with training_args.main_process_first():
        if training_args.do_train:
            train_dataset = process_dataset(raw_datasets['train'])
        if training_args.do_eval:
            eval_dataset = process_dataset(raw_datasets['validation'])
    return train_dataset, eval_dataset, num_labels


def prepare_model(args, training_args, num_labels):
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
        print('This run is for quick check. We will train the model without pretrained '
              f'weights on {quick_check_num} training samples only.')
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    if training_args.seed is not None:
        set_seed(training_args.seed)

    train_dataset, eval_dataset, num_labels = prepare_dataset(args, training_args)
    model = prepare_model(args, training_args, num_labels)

    # wrap with nncf if specified
    compression_ctrl = None
    if args.nncf_config is not None:
        nncf_config = NNCFConfig.from_json(args.nncf_config)
        if nncf_config.get('log_dir', None) is None:
            nncf_config['log_dir'] = training_args.output_dir
        Path(nncf_config['log_dir']).mkdir(parents=True, exist_ok=True)
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
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
