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
import json
import random
import re
import shutil
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import datasets
import numpy as np
import torch
import transformers
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.models.optimum_lm import OptimumLM
from optimum.exporters.openvino.convert import export_from_model
from optimum.intel.openvino import OVModelForCausalLM
from torch import Tensor
from torch import nn
from torch.jit import TracerWarning
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup

import nncf
from nncf.common.logging.track_progress import track
from nncf.data.dataset import Dataset
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.parameters import StripFormat
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.quantize_model import compress_weights
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.model_creation import load_from_config
from nncf.torch.quantization.layers import AsymmetricLoraNLSQuantizer
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import SymmetricLoraNLSQuantizer
from nncf.torch.quantization.layers import SymmetricLoraQuantizer

warnings.filterwarnings("ignore", category=TracerWarning)


def set_trainable(model: nn.Module, lora_lr: float, fq_lr: float) -> list[dict[str, Any]]:
    """
    Sets the trainable parameters of the model for quantization-aware training with LoRA (Low-Rank Adaptation).

    This function disables gradients for all parameters in the model, then selectively enables gradients for
    specific quantizers (AsymmetricLoraQuantizer, SymmetricLoraQuantizer) that have 4-bit quantization.
    It collects the trainable parameters and adapters from these quantizers and returns them in a format
    suitable for an optimizer.

    :param model: The model to be trained.
    :param lora_lr: Learning rate for the LoRA adapters.
    :param fq_lr: Learning rate for the quantizer scales.
    :return: A list of dictionaries containing the parameters to be optimized and their corresponding learning rates.
    """
    model.requires_grad_(False)
    scales_to_train = []
    adapters_to_train = []
    hook_storage = get_hook_storage(model)

    for _, module in hook_storage.named_hooks():
        if isinstance(module, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer)) and (module.num_bits == 4):
            module.enable_gradients()
            params = module.get_trainable_params()
            adapters = module.get_adapters()
            adapters_to_train.extend(adapters.values())
            scales_to_train.extend(param for name, param in params.items() if name not in adapters)

    params = list(model.parameters())
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    all_param = sum(p.numel() for p in params)
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )
    model.train()
    return [{"params": adapters_to_train, "lr": lora_lr}, {"params": scales_to_train, "lr": fq_lr}]


def save_checkpoint(model: nn.Module, ckpt_file: Path) -> None:
    """
    Saves the state of a tuned model from a checkpoint.

    :param model: The model to load the checkpoint into.
    :param ckpt_file: Path to the checkpoint file.
    """
    hook_storage = get_hook_storage(model)
    ckpt = {"nncf_state_dict": hook_storage.state_dict(), "nncf_config": nncf.torch.get_config(model)}
    torch.save(ckpt, ckpt_file)


def load_checkpoint(model: nn.Module, ckpt_file: Path) -> nn.Module:
    """
    Loads the state of a tuned model from a checkpoint. This function restores the placement of Fake Quantizers (FQs)
    with absorbable LoRA adapters and loads their parameters.

    :param model: The model to load the checkpoint into.
    :param ckpt_file: Path to the checkpoint file.
    :returns: The model with the loaded NNCF state from checkpoint.
    """
    ckpt = torch.load(ckpt_file, weights_only=False, map_location="cpu")
    model = load_from_config(model, ckpt["nncf_config"])
    hook_storage = get_hook_storage(model)
    hook_storage.load_state_dict(ckpt["nncf_state_dict"])
    return model


def get_gsm8k() -> list[str]:
    """
    Loads and processes the GSM8K dataset.

    This function loads the GSM8K dataset, processes each sample to extract relevant fields,
    and formats the data into prompts suitable for training.

    :return: A list of processed prompts from the GSM8K dataset.
    """
    train_dataset = datasets.load_dataset("gsm8k", "main", split="train")
    processed_train_dataset = []
    for sample in train_dataset:
        prompt = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
        processed_train_dataset.append(prompt)

    return processed_train_dataset


def get_hellaswag() -> list[str]:
    """
    Loads and processes the HellaSwag dataset.

    :return: A list of processed prompts from the HellaSwag dataset.
    """

    def preprocess(text):
        """Preprocess the text by removing unwanted characters and formatting."""
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    train_dataset = datasets.load_dataset("hellaswag", split="train")
    processed_train_dataset = []
    for sample in train_dataset:
        context = sample["ctx_a"] + " " + sample["ctx_b"].capitalize()
        document = {
            "query": preprocess(sample["activity_label"] + ": " + context),
            "choices": [preprocess(ending) for ending in sample["endings"]],
            "gold": int(sample["label"]),
        }
        query = document["query"]
        answer = document["choices"][document["gold"]]
        prompt = query + " " + answer
        processed_train_dataset.append(prompt)

    return processed_train_dataset


def get_openbookqa() -> list[str]:
    """
    Loads and processes the OpenBookQA dataset.

    :return: A list of processed prompts from the OpenBookQA dataset.
    """
    train_dataset = datasets.load_dataset("openbookqa", split="train")
    processed_train_dataset = []
    for sample in train_dataset:
        document = {
            "id": sample["id"],
            "query": sample["question_stem"],
            "choices": sample["choices"]["text"],
            "gold": ["A", "B", "C", "D"].index(sample["answerKey"].strip()),
        }
        prompt = document["query"]
        answer = document["choices"][document["gold"]]
        prompt = prompt + " " + answer
        processed_train_dataset.append(prompt)

    return processed_train_dataset


def get_winogrande() -> list[str]:
    """
    Loads and processes the Winogrande dataset.

    :return: A list of processed prompts from the Winogrande dataset.
    """
    train_dataset = datasets.load_dataset("winogrande", "winogrande_debiased", split="train")
    processed_train_dataset = []
    for sample in train_dataset:
        pronoun_location = sample["sentence"].index("_")
        answer = sample["option" + sample["answer"]]
        prompt = sample["sentence"][:pronoun_location] + answer + sample["sentence"][pronoun_location + 1 :]
        processed_train_dataset.append(prompt)

    return processed_train_dataset


def get_arc(name: str = "ARC-Easy") -> list[str]:
    """
    Loads and processes the ARC (ARC-Easy or ARC-Challenge) dataset.

    :return: A list of processed prompts from the ARC dataset.
    """
    train_dataset = datasets.load_dataset("ai2_arc", name, split="train")
    processed_train_dataset = []
    for sample in train_dataset:
        # Map numeric answer keys to letter representations.
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        sample["answerKey"] = num_to_letter.get(sample["answerKey"], sample["answerKey"])

        # Process the ARC document to extract relevant fields.
        processed_document = {
            "id": sample["id"],
            "query": "Question: " + sample["question"] + "\nAnswer:",
            "choices": sample["choices"]["text"],
            "gold": ["A", "B", "C", "D", "E"].index(sample["answerKey"]),
        }

        # Construct the prompt with the correct answer.
        answer = processed_document["choices"][processed_document["gold"]]
        prompt = processed_document["query"] + " " + answer
        processed_train_dataset.append(prompt)

    return processed_train_dataset


def lm_eval(
    model: nn.Module,
    task: str,
    batch_size: int = 1,
    tokenizer: AutoTokenizer = None,
) -> dict[str, any]:
    """
    Evaluates a language model on a specified task using the lm-eval library.

    This function initializes a HFLM (from lm-eval) with the provided model and tokenizer,
    and then evaluates it on the specified task.

    :param model: The language model to be evaluated.
    :param tokenizer: The tokenizer corresponding to the language model.
    :param task: The evaluation tasks or task configs.
    :param batch_size: The batch size to be used during evaluation.
    :return: A dictionary containing the evaluation results.
    """
    if isinstance(model, OVModelForCausalLM):
        lm_obj = OptimumLM(pretrained=model, batch_size=batch_size)
    else:
        lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = evaluator.simple_evaluate(lm_obj, tasks=task, log_samples=False)["results"]
    return results[task]


def tokenize(
    tokenizer: AutoTokenizer,
    prompt: str,
    add_eos_token: bool = True,
    max_length: int = 256,
) -> dict[str, list[int]]:
    """
    Tokenize the given prompt.

    :param tokenizer: The tokenizer to use.
    :param prompt: The prompt to tokenize.
    :param add_eos_token: Whether to add an eos token.
    :param max_length: The maximum length of the tokenized input.
    :return: A dictionary containing tokenized input ids, attention mask, and labels.
    """
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors=None,
    )
    if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < max_length and add_eos_token:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result


def get_layer_id_vs_lora_quantizers_map(
    model: nn.Module,
) -> dict[int, list[Union["AsymmetricLoraNLSQuantizer", "SymmetricLoraNLSQuantizer"]]]:
    """
    Maps layer IDs to their corresponding LoRA quantizers.

    :param model: The model containing LoRA quantizers.
    :return: A dictionary mapping layer IDs to lists of LoRA quantizers.
    """
    hook_storage = get_hook_storage(model)
    layer_id_vs_lora_quantizers_map = defaultdict(list)

    for name, module in hook_storage.named_hooks():
        if isinstance(module, (AsymmetricLoraNLSQuantizer, SymmetricLoraNLSQuantizer)) and (module.num_bits == 4):
            match = re.search(r"layers:(\d+):", name)
            if match is None:
                msg = (
                    "Model is supposed to have a specific structure with Transformer blocks "
                    "stored as follows: self.layers = nn.ModuleList(...)"
                )
                raise ValueError(msg)
            layer_id = int(match.group(1))
            layer_id_vs_lora_quantizers_map[layer_id].append(module)

    return layer_id_vs_lora_quantizers_map


@torch.no_grad()
def configure_lora_adapters(
    layer_id_vs_lora_quantizers_map: dict[int, list[Union["AsymmetricLoraNLSQuantizer", "SymmetricLoraNLSQuantizer"]]],
    lora_rank_space: list[int] = None,
    adapter_strategy: str = None,
    specific_rank_config: list[int] = None,
) -> list[int]:
    """
    Configures sub-adapters with specified ranks (or adapter strategy) for each layer in the model.

    :param layer_id_vs_lora_quantizers_map: A dictionary mapping layer IDs to lists of LoRA quantizers.
    :param lora_rank_space: A list of possible ranks for the LoRA adapters.
    :param adapter_strategy: Strategy to select the rank from the `lora_rank_space`.
        Options are 'maximal', 'median', 'minimal', 'random'.
    :param specific_rank_config: A specific configuration of ranks for each layer.
    :return: A list of activated ranks for each layer.
    """
    # Ensure that either [`lora_rank_space` and `adapter_strategy`] or [`specific_rank_config`] is provided
    if specific_rank_config is None:
        assert lora_rank_space and adapter_strategy, (
            "`specific_rank_config` is not provided, both `lora_rank_space` and `adapter_strategy` must be specified."
        )
    else:
        assert len(specific_rank_config) == len(layer_id_vs_lora_quantizers_map), (
            "Length of specific_rank_config must match the number of layers."
        )

    activated_rank_config = []
    for layer, lora_quantizers in layer_id_vs_lora_quantizers_map.items():
        if specific_rank_config is not None:
            selected_rank = specific_rank_config[layer]
        else:
            if adapter_strategy == "maximal":
                selected_rank = lora_rank_space[0]
            elif adapter_strategy == "median":
                selected_rank = lora_rank_space[(len(lora_rank_space) - 1) // 2]
            elif adapter_strategy == "minimal":
                selected_rank = lora_rank_space[-1]
            elif adapter_strategy == "random":
                selected_rank = int(np.random.choice(lora_rank_space))
            else:
                error_message = "Invalid adapter strategy"
                raise ValueError(error_message)

        # Activate the sub-adapter with the selected rank
        for lora_quantizer in lora_quantizers:
            lora_quantizer.set_active_rank(selected_rank)
        activated_rank_config.append(selected_rank)

    return activated_rank_config


@torch.no_grad()
def export_to_openvino(
    pretrained: str,
    ckpt_file: Path,
    ir_dir: Path,
    specific_rank_config: list[int] = None,
) -> OVModelForCausalLM:
    """
    Create a wrapper of OpenVINO model from the checkpoint for evaluation on CPU.

    :param pretrained: The name or path of the pretrained model.
    :param ckpt_file: The path to the checkpoint file to load the model weights and NNCF configurations.
    :param ir_dir: The directory where the OpenVINO model will be saved.
    :param specific_rank_config: A specific configuration of ranks for each layer (only needed if NLS is enabled).
    :return: A wrapper of OpenVINO model ready for evaluation.
    """
    model_to_eval = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch.float32, device_map="cpu")
    example_input = model_to_eval.dummy_inputs
    model_to_eval = load_checkpoint(model_to_eval, ckpt_file)
    if specific_rank_config is not None:
        configure_lora_adapters(
            get_layer_id_vs_lora_quantizers_map(model_to_eval),
            specific_rank_config=specific_rank_config,
        )
    model_to_eval = nncf.strip(model_to_eval, do_copy=False, strip_format=StripFormat.DQ, example_input=example_input)
    export_from_model(model_to_eval, ir_dir, device="cpu")
    return OVModelForCausalLM.from_pretrained(
        model_id=ir_dir,
        trust_remote_code=True,
        load_in_8bit=False,
        compile=True,
        ov_config={"KV_CACHE_PRECISION": "f16", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"},
    )


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)

    # Model params
    parser.add_argument(
        "--pretrained",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="The model id or path of a pretrained HF model configuration.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="output",
        help="Path to the directory for storing logs, tuning checkpoint, compressed model, validation references.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to start from previously saved checkpoint. If not specified or checkpoint does not exist, "
        "start from scratch by post-training weight compression initialization.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Whether to perform evaluation only. If specified, the model will be loaded from the checkpoint.",
    )

    # Downstream task
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "openbookqa",
            "winogrande",
            "arc_challenge",
            "arc_easy",
            "gsm8k",
            "hellaswag",
        ],
        default="openbookqa",
        help="Evaluation task",
    )
    parser.add_argument(
        "--lm_eval_metric",
        type=str,
        default="acc_norm,none",
        help="The metrics of the lm-eval task. Different tasks have different metrics.",
    )

    # Training params
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning. "
        "For larger models (over 2 billion parameters), a learning rate of 5e-4 is recommended.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Size of training batch.")
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=16,
        help="Size of each training microbatch. Gradients will be accumulated until the batch size is reached.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Size of batch for evaluation.")

    # Neural Low-rank Adapter Search (NLS) params
    parser.add_argument(
        "--lora_rank_space",
        type=int,
        nargs="+",
        default=[32, 24, 16],
        help="Search space for LoRA adapter ranks. For example, if the (maximum) rank is 32, "
        "this can be [32, 24, 16] to specify the ranks to be used during NLS.",
    )
    parser.add_argument(
        "--custom_rank_config",
        type=int,
        nargs="+",
        default=None,
        help="Custom LoRA rank configuration (NLS) for evaluation.",
    )
    return parser


def main(argv) -> float:
    """
    Fine-tuning and evaluating a language model with quantization-aware training and LoRA adapters,
    including optional Neural Low-rank Adapter Search (NLS).
    """
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    assert torch.cuda.is_available()
    transformers.set_seed(42)
    device = "cuda"
    torch_dtype = torch.bfloat16
    lora_rank = max(args.lora_rank_space)
    disable_nls = len(args.lora_rank_space) == 1
    do_train = not args.eval_only
    compression_format = CompressionFormat.FQ_LORA if disable_nls else CompressionFormat.FQ_LORA_NLS
    compression_config = dict(
        mode=CompressWeightsMode.INT4_ASYM,
        group_size=64,
        compression_format=compression_format,
        advanced_parameters=AdvancedCompressionParameters(lora_adapter_rank=lora_rank),
    )

    # Configure output and log files.
    output_dir = Path(args.output_dir)
    tensorboard_dir = output_dir / "tb" / datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    last_dir = output_dir / "last"
    ckpt_file = last_dir / "nncf_checkpoint.pth"
    ov_dir = output_dir / "ov"
    result_file = output_dir / "result.json"

    if not do_train:
        assert args.resume and ckpt_file.exists(), (
            "Only supports evaluating trained models when do_train is False. "
            "Please enable --resume and ensure that a checkpoint exists in output_dir/last."
        )
        assert disable_nls or args.custom_rank_config is not None, "Please provide `custom_rank_config` for evaluation."

    if not args.resume:
        shutil.rmtree(output_dir, ignore_errors=True)
    for path in [output_dir, tensorboard_dir, last_dir]:
        path.mkdir(exist_ok=True, parents=True)
    print(f"To visualize the loss, open Tensorboard using the logs from: {tensorboard_dir}")
    tb = SummaryWriter(tensorboard_dir, "QAT with absorbable LoRA")
    overall_result = {}

    # Load original model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(args.pretrained, torch_dtype=torch_dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results_before_compression = lm_eval(model, task=args.task, batch_size=args.eval_batch_size, tokenizer=tokenizer)
    print(f"Results before compression={json.dumps(results_before_compression, indent=4)}")
    overall_result["results_before_compression"] = results_before_compression

    # Dataset preparation
    train_dataset = None
    if args.task == "gsm8k":
        train_dataset = get_gsm8k()
    elif args.task == "hellaswag":
        train_dataset = get_hellaswag()
    elif args.task == "openbookqa":
        train_dataset = get_openbookqa()
    elif args.task == "winogrande":
        train_dataset = get_winogrande()
    elif args.task == "arc_challenge":
        train_dataset = get_arc(name="ARC-Challenge")
    elif args.task == "arc_easy":
        train_dataset = get_arc(name="ARC-Easy")
    else:
        error_message = f"Unsupported task: {args.task}."
        raise ValueError(error_message)
    model_input = model.dummy_inputs
    train_dataset = [tokenize(tokenizer, sample) for sample in train_dataset]
    random.shuffle(train_dataset)

    model = compress_weights(
        model,
        dataset=Dataset([{k: v.to(device) for k, v in model_input.items()}]),
        **compression_config,
    )
    results_of_compressed_model = lm_eval(model, task=args.task, batch_size=args.eval_batch_size, tokenizer=tokenizer)
    print(f"Results of NNCF compressed model={json.dumps(results_of_compressed_model, indent=4)}")
    overall_result["results_of_compressed_model"] = results_of_compressed_model
    initial_result = results_of_compressed_model[args.lm_eval_metric]

    # Create or load model to tune with Fake Quantizers and absorbable LoRA adapters.
    if args.resume and ckpt_file.exists():
        model = AutoModelForCausalLM.from_pretrained(args.pretrained, torch_dtype=torch_dtype, device_map=device)
        model = load_checkpoint(model, ckpt_file)
    else:
        save_checkpoint(model, ckpt_file)

    layer_id_vs_lora_quantizers_map = None
    if not disable_nls:
        layer_id_vs_lora_quantizers_map = get_layer_id_vs_lora_quantizers_map(model)

    if do_train:
        fq_lr = args.lr / 10
        weight_decay = args.lr
        param_to_train = set_trainable(model, lora_lr=args.lr, fq_lr=fq_lr)
        opt = torch.optim.AdamW(param_to_train, weight_decay=weight_decay)

        grad_accumulation_steps = args.batch_size // args.microbatch_size
        num_samples = len(train_dataset)
        epoch_samples = num_samples - num_samples % args.microbatch_size
        microbatches_per_epoch = epoch_samples // args.microbatch_size
        aggregated_loss = float("nan")
        loss_numerator = grad_steps = total_microbatches = 0
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
        total_steps = (microbatches_per_epoch * args.epochs) // grad_accumulation_steps
        scheduler = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )

        if disable_nls:
            activation_counter = None
            loss_recorder = None
        else:
            # Initialize the counter for tracking activation counts during training
            maximal_lora_rank_config = configure_lora_adapters(
                layer_id_vs_lora_quantizers_map,
                lora_rank_space=args.lora_rank_space,
                adapter_strategy="maximal",
            )
            activation_counter = [
                {rank: 0 for rank in args.lora_rank_space} for _ in range(len(maximal_lora_rank_config))
            ]

            # Initialize the loss recorder for tracking losses during training (for each sub-adapter)
            loss_recorder = defaultdict(list)

        for epoch in range(args.epochs):
            batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
            for indices in track(batch_indices_epoch, description=f"Train epoch {epoch}"):
                # If Neural Low-rank Adapter Search (NLS) is enabled,
                # configure the LoRA adapters with a random rank configuration from the specified rank space.
                if not disable_nls and grad_steps == 0:
                    current_config = configure_lora_adapters(
                        layer_id_vs_lora_quantizers_map,
                        lora_rank_space=args.lora_rank_space,
                        adapter_strategy="random",
                    )
                    # Update the activation counter
                    for idx, rank in enumerate(current_config):
                        activation_counter[idx][rank] += 1
                    current_config_tuple = tuple(current_config)

                indices = indices.tolist()
                total_microbatches += 1

                def form_batch(inputs: list[Tensor]):
                    batch = [inputs[i] for i in indices]
                    batch = data_collator(batch)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    return batch

                inputs = form_batch(train_dataset)
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

                # Record the loss for the current configuration
                if not disable_nls:
                    loss_recorder[current_config_tuple].append(loss.item())

                # Perform an optimization step after accumulating gradients over multiple minibatches.
                loss_numerator += loss.item()
                grad_steps += 1
                if not torch.isfinite(loss).item():
                    err = f"Fine-tuning loss is {loss}"
                    raise ValueError(err)
                (loss / grad_accumulation_steps).backward()
                if grad_steps == grad_accumulation_steps:
                    opt.step()
                    scheduler.step()
                    opt.zero_grad()
                    aggregated_loss = loss_numerator / grad_steps
                    loss_numerator = grad_steps = 0

                    current_lr = scheduler.get_last_lr()[0]
                    if total_microbatches % 10 == 0:
                        print(
                            f"Epoch: {epoch + 1}, "
                            f"Step: {total_microbatches}, "
                            f"Loss: {aggregated_loss:.4f}, "
                            f"Learning Rate: {current_lr:.6f}"
                        )
                    tb.add_scalar("learning_rate", current_lr, total_microbatches)
                    tb.add_scalar("loss", aggregated_loss, total_microbatches)

            save_checkpoint(model, ckpt_file)

    # Start evaluation
    if disable_nls:
        results_of_lora_finetuned_compressed_model = lm_eval(
            model, task=args.task, batch_size=args.eval_batch_size, tokenizer=tokenizer
        )
        print(
            f"Results of quantization-aware-finetuned (LoRA) NNCF compressed model="
            f"{json.dumps(results_of_lora_finetuned_compressed_model, indent=4)}"
        )
        overall_result["lora_results"] = results_of_lora_finetuned_compressed_model
        best_result = results_of_lora_finetuned_compressed_model[args.lm_eval_metric]
    else:
        overall_result["nls_results"] = []
        # Use some of the signals from training to find some heuristic configurations for evaluation.
        if do_train:
            # Extract the most frequently activated configuration
            def get_most_frequent_config(activation_counter):
                most_frequent_config = []
                for layer_counter in activation_counter:
                    most_frequent_rank = max(layer_counter, key=layer_counter.get)
                    most_frequent_config.append(most_frequent_rank)
                return most_frequent_config

            # Calculate the average loss for each configuration and select the top k with the minimum loss
            def get_top_k_min_loss_configs(loss_recorder, k=5):
                avg_loss_configs = [(config, sum(losses) / len(losses)) for config, losses in loss_recorder.items()]
                avg_loss_configs.sort(key=lambda x: x[1])
                top_k_configs = [list(config) for config, _ in avg_loss_configs[:k]]
                return top_k_configs

            best_result = float("-inf")
            best_config = None

            # Test the median configuration
            median_lora_rank_config = configure_lora_adapters(
                layer_id_vs_lora_quantizers_map,
                lora_rank_space=args.lora_rank_space,
                adapter_strategy="median",
            )
            results_of_nls_finetuned_compressed_model_median = lm_eval(
                model, task=args.task, batch_size=args.eval_batch_size, tokenizer=tokenizer
            )
            print(
                f"Results of quantization-aware-finetuned (NLS-Median) NNCF compressed model="
                f"{json.dumps(results_of_nls_finetuned_compressed_model_median, indent=4)}"
            )
            overall_result["nls_results"].append(
                {
                    "type": "median",
                    "config": median_lora_rank_config,
                    "results": results_of_nls_finetuned_compressed_model_median,
                }
            )
            if results_of_nls_finetuned_compressed_model_median[args.lm_eval_metric] > best_result:
                best_result = results_of_nls_finetuned_compressed_model_median[args.lm_eval_metric]
                best_config = median_lora_rank_config

            # Test the most frequent configuration
            most_frequent_lora_rank_config = get_most_frequent_config(activation_counter)
            configure_lora_adapters(
                layer_id_vs_lora_quantizers_map,
                specific_rank_config=most_frequent_lora_rank_config,
            )
            results_of_nls_finetuned_compressed_model_most_frequent = lm_eval(
                model, task=args.task, batch_size=args.eval_batch_size, tokenizer=tokenizer
            )
            print(
                f"Results of quantization-aware-finetuned (NLS-Most-Frequent) NNCF compressed model="
                f"{json.dumps(results_of_nls_finetuned_compressed_model_most_frequent, indent=4)}"
            )
            overall_result["nls_results"].append(
                {
                    "type": "most-frequent",
                    "config": most_frequent_lora_rank_config,
                    "results": results_of_nls_finetuned_compressed_model_most_frequent,
                }
            )
            if results_of_nls_finetuned_compressed_model_most_frequent[args.lm_eval_metric] > best_result:
                best_result = results_of_nls_finetuned_compressed_model_most_frequent[args.lm_eval_metric]
                best_config = most_frequent_lora_rank_config

            # Test the top 5 min loss configurations
            top_5_min_loss_configs = get_top_k_min_loss_configs(loss_recorder, k=5)
            for i, min_loss_config in enumerate(top_5_min_loss_configs):
                configure_lora_adapters(
                    layer_id_vs_lora_quantizers_map,
                    specific_rank_config=min_loss_config,
                )
                results_of_nls_finetuned_compressed_model_min_loss = lm_eval(
                    model, task=args.task, batch_size=args.eval_batch_size, tokenizer=tokenizer
                )
                print(
                    f"Results of quantization-aware-finetuned (NLS-Min-Loss-{i + 1}) NNCF compressed model="
                    f"{json.dumps(results_of_nls_finetuned_compressed_model_min_loss, indent=4)}"
                )
                overall_result["nls_results"].append(
                    {
                        "type": f"min-loss-{i + 1}",
                        "config": min_loss_config,
                        "results": results_of_nls_finetuned_compressed_model_min_loss,
                    }
                )
                if results_of_nls_finetuned_compressed_model_min_loss[args.lm_eval_metric] > best_result:
                    best_result = results_of_nls_finetuned_compressed_model_min_loss[args.lm_eval_metric]
                    best_config = min_loss_config
        else:
            configure_lora_adapters(
                layer_id_vs_lora_quantizers_map,
                specific_rank_config=args.custom_rank_config,
            )
            results_of_nls_finetuned_compressed_model_custom = lm_eval(
                model, task=args.task, batch_size=args.eval_batch_size, tokenizer=tokenizer
            )
            print(
                f"Results of quantization-aware-finetuned (NLS with custom config) NNCF compressed model="
                f"{json.dumps(results_of_nls_finetuned_compressed_model_custom, indent=4)}"
            )
            overall_result["nls_results"].append(
                {
                    "type": "custom",
                    "config": args.custom_rank_config,
                    "results": results_of_nls_finetuned_compressed_model_custom,
                }
            )
            best_config = args.custom_rank_config
        overall_result["nls_best_config"] = best_config

    if disable_nls:
        ov_model = export_to_openvino(args.pretrained, ckpt_file, ov_dir)
    else:
        ov_model = export_to_openvino(args.pretrained, ckpt_file, ov_dir, best_config)
    ov_result = lm_eval(
        ov_model,
        task=args.task,
        batch_size=args.eval_batch_size,
    )
    overall_result["ov_result"] = ov_result
    print(f"Overall result: {json.dumps(overall_result, indent=4)}")

    # Save results
    with open(result_file, "w") as f:
        json.dump(overall_result, f, indent=4)

    print(f"The finetuned model has been exported to OpenVINO and saved at: {ov_dir.resolve()}")
    print(f"Results have been saved to: {result_file.resolve()}")

    best_result = ov_result[args.lm_eval_metric]
    tb.add_scalar("ov_results", best_result, 0)
    result_diff = best_result - initial_result
    result_diff = round(result_diff, 2)
    return result_diff


if __name__ == "__main__":
    main(sys.argv[1:])
