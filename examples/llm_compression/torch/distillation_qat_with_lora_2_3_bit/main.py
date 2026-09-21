# Copyright (c) 2026 Intel Corporation
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
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from optimum.exporters.openvino.convert import export_from_model
from optimum.intel.openvino import OVModelForCausalLM
from torch import Tensor
from torch import nn
from torch.jit import TracerWarning
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.common.logging.track_progress import track
from nncf.data.dataset import Dataset
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.parameters import StripFormat
from nncf.quantization.advanced_parameters import AdvancedAWQParameters
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.quantize_model import compress_weights
from nncf.quantization.quantize_model import repack_weights
from nncf.torch import load_from_config
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import SymmetricLoraQuantizer

warnings.filterwarnings("ignore", category=TracerWarning)


def get_wikitext2(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device) -> list[Tensor]:
    """
    Loads and processes the Wikitext-2 dataset for training.

    :param num_samples: Number of samples to generate.
    :param seqlen: Sequence length for each sample.
    :param tokenizer: Tokenizer to encode the text.
    :param device: Device to move the tensors to (e.g., 'cpu' or 'cuda').
    :return: A list of tensors containing the tokenized text samples.
    """
    traindata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    limit = num_samples * seqlen  # ~1k for 128 samples with seqlen=32 to be aligned with optimum
    text = "".join([" \n" if s == "" else s for s in traindata["text"][:limit]])
    trainenc = tokenizer(text, return_tensors="pt")
    trainloader = []
    for _ in range(num_samples):
        # Crop a sequence of tokens of length seqlen starting at a random position
        i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        trainloader.append(inp)
    return trainloader


def get_ultrachat_200k(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device) -> list[Tensor]:
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.apply_chat_template is None:
        msg = "Tokenizer must have an 'apply_chat_template' attribute for ultra chat dataset."
        raise ValueError(msg)

    trainloader = []
    text = ""

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)

    for example in dataset:
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
        trainenc = tokenizer(text, return_tensors="pt")

        if trainenc.input_ids.shape[1] < seqlen:
            continue
        if trainenc.input_ids.shape[1] > seqlen + 1:
            i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        else:
            i = 0
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        trainloader.append(inp)
        if len(trainloader) >= num_samples:
            break
    del dataset

    return trainloader


def get_pile_10k(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device) -> list[Tensor]:
    ds = load_dataset("NeelNanda/pile-10k", split="train")

    trainloader = []
    text = ""
    for example in ds:
        text += " \n" + example["text"]
        trainenc = tokenizer(text, return_tensors="pt")

        if trainenc.input_ids.shape[1] < seqlen:
            continue
        text = ""
        if trainenc.input_ids.shape[1] > seqlen + 1:
            i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        else:
            i = 0
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        trainloader.append(inp)
        if len(trainloader) >= num_samples:
            break

    return trainloader


@torch.no_grad()
def calc_hiddens(model: nn.Module, dataloader: list[Tensor]) -> list[Tensor]:
    """
    Calculate the hidden states for each input in the dataloader using the given model.

    :param model: The model used to calculate the hidden states.
    :param dataloader: The dataloader providing the inputs to the model.
    :return: A list of hidden states for each input in the dataloader.
    """
    orig_hiddens = []
    for data in track(dataloader, description="Calculating original hiddens"):
        model_input = get_model_input(data)
        orig_hiddens.append(model.model(**model_input).last_hidden_state.cpu())
    torch.cuda.empty_cache()
    return orig_hiddens


def get_model_input(input_ids: Tensor) -> dict[str, Tensor]:
    """
    Prepares the model input dictionary with input IDs, attention mask, and position IDs.

    :param input_ids: Tensor containing the input IDs.
    :return: A dictionary with keys "input_ids", "attention_mask", and "position_ids",
        each mapping to their respective tensors.
    """
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.cumsum(attention_mask, axis=1) - 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}


def kl_div(student_hiddens: torch.Tensor, teacher_hiddens: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kullback-Leibler divergence loss between the student and teacher hidden states.
    The input tensors are expected to have the same shape, and the last dimension represents the number of classes.

    :param student_hiddens: The hidden states from the student model.
    :param teacher_hiddens: The hidden states from the teacher model.
    :returns: The computed KL divergence loss.
    """
    num_classes = student_hiddens.shape[-1]
    return F.kl_div(
        input=F.log_softmax(student_hiddens.view(-1, num_classes), dim=-1),
        target=F.log_softmax(teacher_hiddens.view(-1, num_classes), dim=-1),
        log_target=True,
        reduction="batchmean",
    )


def set_trainable(model: nn.Module, lora_lr: float, fq_lr: float, scales_only: bool = False) -> list[dict[str, Any]]:
    """
    Sets the trainable parameters of the model for quantization-aware training with LoRA (Low-Rank Adaptation).

    This function disables gradients for all parameters in the model, then selectively enables gradients for
    specific quantizers (AsymmetricLoraQuantizer, SymmetricLoraQuantizer) that have 4-bit quantization.
    It collects the trainable parameters and adapters from these quantizers and returns them in a format
    suitable for an optimizer.

    :param model: The model to be trained.
    :param lora_lr: Learning rate for the LoRA adapters.
    :param fq_lr: Learning rate for the quantizer scales.
    :param scales_only: If True, only quantizer scales are trainable; LoRA adapters are frozen.
    :return: A list of dictionaries containing the parameters to be optimized and their corresponding learning rates.
    """
    model.requires_grad_(False)
    scales_to_train = []
    adapters_to_train = []
    hook_storage = get_hook_storage(model)
    for _, module in hook_storage.named_hooks():
        if isinstance(module, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer)) and (module.num_bits < 4):
            module.enable_gradients()
            params = module.get_trainable_params()
            adapters = module.get_adapters()
            if scales_only:
                for adapter in adapters.values():
                    adapter.requires_grad_(False)
            else:
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
    if scales_only:
        return [{"params": scales_to_train, "lr": fq_lr}]
    return [{"params": adapters_to_train, "lr": lora_lr}, {"params": scales_to_train, "lr": fq_lr}]


def get_linear_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    enabled: bool,
    num_epochs: int,
    microbatches_per_epoch: int,
    grad_accumulation_steps: int,
) -> LinearLR | None:
    """
    Creates a linear learning rate scheduler for the requested training phase.

    :param optimizer: Optimizer to schedule.
    :param enabled: Whether to create the scheduler.
    :param num_epochs: Number of epochs in the phase.
    :param microbatches_per_epoch: Number of microbatch chunks in one epoch.
    :param grad_accumulation_steps: Number of microbatches before an optimizer step.
    :return: A linear learning rate scheduler or None.
    """
    if not enabled:
        return None
    total_steps = num_epochs * (microbatches_per_epoch // grad_accumulation_steps)
    if total_steps == 0:
        return None
    return LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)


def save_checkpoint(model: nn.Module, ckpt_file: Path, model_state: bool = True) -> None:
    """
    Stores the current state of a quantized model to a checkpoint file.

    :param model: The model whose state will be saved to checkpoint.
    :param ckpt_file: Path to store the checkpoint file.
    :param model_state: Whether to save the complete model weights in addition to NNCF state. Required when using
        AWQ method which fuses scaling factors into weights. When False, only NNCF configuration and state are saved,
        as they're maintained separately from the model's weights.
    """
    hook_storage = get_hook_storage(model)
    ckpt = {"nncf_state_dict": hook_storage.state_dict(), "nncf_config": nncf.torch.get_config(model)}
    if model_state:
        ckpt["model_state"] = model.state_dict()
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
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    hook_storage = get_hook_storage(model)
    hook_storage.load_state_dict(ckpt["nncf_state_dict"])
    return model


@torch.no_grad()
def export_to_openvino(pretrained: str, ckpt_file: Path, ir_dir: Path):
    """
    Export the quantized model to OpenVINO IR format.

    :param pretrained: The name or path of the pretrained model.
    :param ckpt_file: The path to the checkpoint file to load the model weights and NNCF configurations.
    :param ir_dir: The directory where the OpenVINO model will be saved.
    :return: A wrapper of OpenVINO model ready for evaluation.
    """
    model_to_eval = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch.float32, device_map="cpu")
    model_to_eval = load_checkpoint(model_to_eval, ckpt_file)
    model_to_eval = nncf.strip(model_to_eval, do_copy=False, strip_format=StripFormat.DQ)
    export_from_model(model_to_eval, ir_dir, device="cpu")
    print(f"The OpenVINO model has been exported and saved to: {ir_dir}")

    model_to_eval = OVModelForCausalLM.from_pretrained(ir_dir)
    model_to_eval.model = repack_weights(model_to_eval.model)
    model_to_eval.save_pretrained(ir_dir / "repacked")
    print(f"The OpenVINO model has been repacked and saved to: {ir_dir / 'repacked'}")


@torch.no_grad()
def export_to_dequantized_torch(pretrained: str, ckpt_file: Path, pt_dir: Path):
    """
    Replace the quantized weights of the model with dequantized weights and save it to the specified directory.

    :param pretrained: The name or path of the pretrained model.
    :param ckpt_file: The path to the checkpoint file to load the model weights and NNCF configurations.
    :param pt_dir: The directory where the dequantized PyTorch model will be saved.
    :return: None. The dequantized model is saved to the specified directory.
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model_to_eval = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch.float32, device_map="cpu")
    model_to_eval = load_checkpoint(model_to_eval, ckpt_file)
    model_to_eval = nncf.strip(model_to_eval, do_copy=False, strip_format=StripFormat.IN_PLACE)
    model_to_eval.save_pretrained(pt_dir)
    tokenizer.save_pretrained(pt_dir)


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)

    # Model params
    parser.add_argument(
        "--pretrained",
        type=str,
        # default="Qwen/Qwen3-8B",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="The model id or path of a pretrained HF model configuration.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=3,
        choices=[2, 3],
        help="Number of bits for weight compression (2 or 3).",
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
    parser.add_argument("--lora_rank", type=int, default=256, help="Rank of lora adapters")
    parser.add_argument(
        "--basic_init",
        action="store_true",
        help="Whether to initialize quantization with basic min-max round-to-nearest schema. By default, advanced "
        "data-aware post-training methods are used: AWQ + Scale Estimation. These methods typically provide better "
        "accuracy, but require a calibration dataset and additional initialization time "
        "(~20 sec for 1B and ~80 sec for 8B models).",
    )
    parser.add_argument(
        "--equalize_mlp",
        action="store_true",
        help="Whether to equalize the scales of MLP layers (down_proj and up_proj/gate_proj with preceding LayerNorm) "
        "before quantization. This can improve the accuracy of low-bit quantization.",
    )
    parser.add_argument(
        "--save_pt",
        action="store_true",
        help="Whether to save the model with dequantization after training. It is useful for fast evaluation.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pile-10k",
        choices=["pile-10k", "wikitext-2", "ultrachat_200k"],
        help="The dataset to use for training and evaluation.",
    )

    # Data params
    parser.add_argument("--num_train_samples", type=int, default=2048, help="Number of training samples")
    parser.add_argument("--train_seqlen", type=int, default=1024, help="Train data context length.")

    # Training params
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning. "
        "For larger models (over 3 billion parameters), a learning rate of 5e-5 is recommended.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument(
        "--scale_epochs",
        type=int,
        default=0,
        help="Number of additional epochs after the main training loop during which only the quantizer scales are "
        "finetuned (LoRA adapters are frozen). Set to 0 to skip this phase.",
    )
    parser.add_argument(
        "--linear_lr_scheduler",
        action="store_true",
        help="Use a linear learning rate scheduler that decays the learning rate to zero during each training phase.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Size of training batch.")
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=8,
        help="Size of each training microbatch. Gradients will be accumulated until the batch size is reached.",
    )
    return parser


# ---------------------------------------------------------------------- #
# MLP equalization (average up_proj/gate_proj input scale absorbed into layer norm weights)
# ---------------------------------------------------------------------- #
def _find_mlp_groups(model: nn.Module) -> list[tuple[nn.Module, nn.Linear, list[nn.Linear]]]:
    """
    Collect ``(parent, down_proj, [producers])`` triples where ``producers``
    are the sibling linears whose outputs are consumed by ``down_proj`` along
    its input-channel dimension.

    Recognized layouts:
      * Llama-style: ``down_proj`` consumes ``up_proj`` * SiLU(``gate_proj``);
        producers = ``[up_proj, gate_proj]``.
      * Generic: only ``up_proj`` present -> producers = ``[up_proj]``.
    """
    groups: list[tuple[nn.Module, nn.Linear, list[nn.Linear]]] = []
    for parent in model.modules():
        down = getattr(parent, "down_proj", None)
        if not isinstance(down, nn.Linear):
            continue
        producers: list[nn.Linear] = []
        for attr in ("up_proj",):
            sib = getattr(parent, attr, None)
            if isinstance(sib, nn.Linear) and sib.out_features == down.in_features:
                producers.append(sib)
        if producers:
            groups.append((parent, down, producers))
    return groups


def _find_up_gate_groups(model: nn.Module) -> list[tuple[nn.Module, nn.Linear, list[nn.Linear]]]:
    """
    Collect ``(parent, down_proj, [producers])`` triples where ``producers``
    are the sibling linears whose outputs are consumed by ``down_proj`` along
    its input-channel dimension.

    Recognized layouts:
      * Llama-style: ``down_proj`` consumes ``up_proj`` * SiLU(``gate_proj``);
        producers = ``[up_proj, gate_proj]``.
      * Generic: only ``up_proj`` present -> producers = ``[up_proj]``.
    """
    groups: list[tuple[nn.Module, nn.Linear, list[nn.Linear]]] = []
    for parent in model.modules():
        if not hasattr(parent, "mlp"):
            continue
        mlp = getattr(parent, "mlp")
        gate = getattr(mlp, "gate_proj", None)
        if not isinstance(gate, nn.Linear):
            continue

        up = getattr(mlp, "up_proj", None)
        if not isinstance(up, nn.Linear):
            continue

        # Identify the LayerNorm whose output is the direct input to gate_proj / up_proj.
        # Check Gemma3-style first (pre_feedforward_layernorm), then fall back to
        # Llama/Qwen-style (post_attention_layernorm).
        producer = None
        for attr in ("pre_feedforward_layernorm", "post_attention_layernorm"):
            sib = getattr(parent, attr, None)
            if sib is not None:
                producer = sib
                break

        if producer:
            groups.append((up, gate, producer))
    return groups


def _get_weight_data(module: nn.Module) -> torch.Tensor | None:
    """Return the actual weight tensor for *module*, even when the parameter is on
    the meta device (i.e. CPU-offloaded by accelerate's device_map='auto').

    For meta parameters, the real data lives in the AlignDevicesHook's
    ``weights_map`` dict-like object; we return a copy on CPU so arithmetic works.
    Returns ``None`` if the actual data cannot be found.
    """
    w = module.weight
    if w.device.type != "meta":
        return w.data
    return None


def _set_weight_data(module: nn.Module, data: torch.Tensor) -> None:
    """Write *data* back into *module*'s weight, even when it is on meta device.

    For non-meta parameters: in-place copy to the parameter's device.
    For meta (offloaded) parameters: replace the entry in the hook's weights_map.
    """
    w = module.weight
    if w.device.type != "meta":
        w.data.copy_(data.to(device=w.device, dtype=w.dtype))


# rescale scale to [min, max] to avoid extreme values that cause instability during training or quantization
def align_scale(s: Tensor, min=0.1, max=1.0) -> Tensor:
    min_s = s.min()
    max_s = s.max()
    if max_s - min_s < 1e-5:
        return torch.clamp(s, min=min, max=max)
    s = (s - min_s) / (max_s - min_s) * (max - min) + min
    return s


@torch.no_grad()
def equalize_up_gate_with_layernorm(model: nn.Module, eps: float = 1e-5, use_align_scale: bool = True) -> int:
    groups = _find_up_gate_groups(model)
    if not groups:
        return 0

    n_done = 0
    for up, gate, producer in groups:
        w_gate = _get_weight_data(gate)
        w_up = _get_weight_data(up)
        w_prod = _get_weight_data(producer)
        if w_gate is None or w_up is None or w_prod is None:
            continue

        s_gate = w_gate.abs().mean(dim=0).clamp_min(eps).to(dtype=w_gate.dtype)
        s_up = w_up.abs().mean(dim=0).clamp_min(eps).to(dtype=w_up.dtype)

        s_gate = s_gate / s_gate.norm(p=2, dim=0, keepdim=True)
        s_up = s_up / s_up.norm(p=2, dim=0, keepdim=True)

        # up_proj theoretically more sensitive to quantization
        s = 0.1 * s_gate + 0.9 * s_up
        if use_align_scale:
            s = align_scale(s, min=0.1, max=1.0)
        # Divide gate/up input columns by s.
        _set_weight_data(gate, w_gate * (1.0 / s.unsqueeze(0)))
        _set_weight_data(up, w_up * (1.0 / s.unsqueeze(0)))

        # Scale producer (LayerNorm/RMSNorm) so its output is multiplied by s.
        s_dev = s.to(dtype=w_prod.dtype)
        # Gemma3 uses (1 + weight) RMSNorm (weight initialized to zeros):
        #   effective multiplier = (1 + weight)
        #   to scale output by s: (1 + w_new) = s*(1 + w_old)  →  w_new = s*(1+w_old) - 1
        # Standard RMSNorm (e.g. Llama) uses weight*x (weight initialized to ones):
        #   effective multiplier = weight
        #   to scale output by s: w_new = s * w_old
        if "gemma" in type(producer).__name__.lower():
            _set_weight_data(producer, s_dev * (1.0 + w_prod) - 1.0)
        else:
            _set_weight_data(producer, w_prod * s_dev)
        if hasattr(producer, "bias") and producer.bias is not None and producer.bias.device.type != "meta":
            producer.bias.data.mul_(s_dev.to(producer.bias.device, producer.bias.dtype))
        n_done += 1
    return n_done


@torch.no_grad()
def equalize_down_proj(
    model: nn.Module,
    eps: float = 1e-5,
    use_align_scale: bool = True,
) -> int:
    """
    Equalize each ``down_proj`` layer by absorbing the per-input-channel
    activation magnitude into its producers (``up_proj`` and, when present,
    ``gate_proj``).

    For every MLP block let ``s = mean(|x|, dim=batch_seq)`` measured at the
    input of ``down_proj`` over the calibration set. Then:

    * ``down_proj.weight  /= s[None, :]`` (divide along input channels)
    * For each producer ``L`` (e.g. ``up_proj``, ``gate_proj``):
      ``L.weight *= s[:, None]``  (scale output channels)
      ``L.bias   *= s``           (if a bias exists)

    Mathematically, ``down(up(x) * silu(gate(x))) = down((up(x)*s) * (silu(gate(x)*s)/s))``
    is *not* exact for the SiLU branch in general, but in practice this
    pre-quantization equalization (cf. SmoothQuant / AWQ) significantly
    flattens the weight magnitudes seen by the per-group quantizer. The
    transformation is exact when no SiLU is present (``producers == [up_proj]``).

    :param model: Model whose MLP blocks expose ``down_proj`` (and optional
        ``up_proj``/``gate_proj`` siblings) as direct attributes. Must be
        called on plain ``nn.Linear`` layers (i.e. **before** wrapping them
        with :class:`QuantizedLoraLinear`).
    :param eps: Lower bound for ``s`` to avoid division by zero.
    :return: Number of equalized MLP groups.
    """
    groups = _find_mlp_groups(model)
    if not groups:
        return 0

    n_done = 0
    for _, down, producers in groups:
        w_down = _get_weight_data(down)
        if w_down is None:
            continue
        s = w_down.abs().mean(dim=0).clamp_min(eps).to(dtype=w_down.dtype)
        if use_align_scale:
            s = align_scale(s, min=0.1, max=1.0)
        _set_weight_data(down, w_down * (1.0 / s.unsqueeze(0)))

        # Scale producer output rows by s.
        for prod in producers:
            w_prod = _get_weight_data(prod)
            if w_prod is None:
                continue
            s_dev = s.to(dtype=w_prod.dtype)
            _set_weight_data(prod, w_prod * s_dev.unsqueeze(1))
            if prod.bias is not None and prod.bias.device.type != "meta":
                prod.bias.data.mul_(s_dev.to(prod.bias.device, prod.bias.dtype))
        n_done += 1
    return n_done


def run_training(
    model: nn.Module,
    train_loader: list[Tensor],
    orig_hiddens: list[Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None,
    ckpt_file: Path,
    tb: SummaryWriter,
    num_epochs: int,
    phase_desc: str,
    start_total_steps: int,
    *,
    device: torch.device | str,
    torch_dtype: torch.dtype,
    grad_accumulation_steps: int,
    num_samples: int,
    epoch_samples: int,
    microbatches_per_epoch: int,
    model_state: bool,
) -> int:
    """
    Run the distillation-based training loop for the compressed model.

    :param model: The model being tuned.
    :param train_loader: Training samples used for distillation.
    :param orig_hiddens: Teacher hidden states computed from the original model.
    :param optimizer: Optimizer used for updates.
    :param scheduler: Optional learning rate scheduler stepped after optimizer updates.
    :param ckpt_file: Path to save the checkpoint after each epoch.
    :param tb: TensorBoard writer for loss metrics.
    :param num_epochs: Number of epochs to run.
    :param phase_desc: Human-readable description of the current epoch phase.
    :param start_total_steps: The initial global optimizer step count.
    :param device: Device to place batch tensors on.
    :param torch_dtype: The dtype used for the computation.
    :param grad_accumulation_steps: Number of microbatches before an optimizer step.
    :param num_samples: Number of training samples.
    :param epoch_samples: The available number of samples in the epoch after rounding.
    :param microbatches_per_epoch: Number of microbatch chunks in the epoch.
    :param model_state: Whether to save full model weights in the checkpoint.
    :return: The total number of optimizer steps performed.
    """
    loss_numerator = grad_steps = 0
    total_steps = start_total_steps
    for epoch in range(num_epochs):
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        epoch_tracker = track(batch_indices_epoch, description=f"{phase_desc} {epoch}")
        for indices in epoch_tracker:
            indices = indices.tolist()

            def form_batch(inputs: list[Tensor], model_input: bool):
                batch = torch.cat([inputs[i] for i in indices], dim=0)
                return get_model_input(batch) if model_input else batch.to(device=device, dtype=torch_dtype)

            # Compute distillation loss between logits of the original model and the model with FQ + LoRA.
            inputs = form_batch(train_loader, model_input=True)
            with torch.no_grad():
                targets = model.lm_head(form_batch(orig_hiddens, model_input=False))
                if hasattr(model.config, "final_logit_softcapping"):  # Gemma has post-processing after lm_head
                    fls = model.config.final_logit_softcapping
                    if fls is not None:
                        targets = targets / fls
                        targets = torch.tanh(targets)
                        targets = targets * fls
            outputs = model(**inputs).logits
            loss = kl_div(outputs, targets.to(dtype=torch_dtype, device=device))

            # Perform an optimization step after accumulating gradients over multiple minibatches.
            loss_numerator += loss.item()
            grad_steps += 1
            if not torch.isfinite(loss).item():
                err = f"Fine-tuning loss is {loss}"
                raise ValueError(err)
            (loss / grad_accumulation_steps).backward()
            if grad_steps == grad_accumulation_steps:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                aggregated_loss = loss_numerator / grad_steps
                loss_numerator = grad_steps = 0
                total_steps += 1
                tb.add_scalar("loss", aggregated_loss, total_steps)
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
                tb.add_scalar("lr", current_lr, total_steps)
                epoch_tracker.update(advance=0, description=f"{phase_desc} {epoch} | loss={aggregated_loss:.4f}")

        save_checkpoint(model, ckpt_file, model_state=model_state)
    return total_steps


def main(argv) -> float:
    """
    Fine-tunes the specified model and returns the difference between initial and best validation perplexity in Torch,
    and the test perplexity for best model exported to OpenVINO.
    """
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    assert torch.cuda.is_available()
    transformers.set_seed(42)
    device = "cuda"
    torch_dtype = torch.bfloat16
    compression_config = dict(
        mode=CompressWeightsMode.INT3_SYM if args.bits == 3 else CompressWeightsMode.INT2_SYM,
        group_size=64,
        awq=not (args.basic_init or args.equalize_mlp),
        scale_estimation=not args.basic_init,
        compression_format=CompressionFormat.FQ_LORA,
    )
    pprint({"CLI arguments": vars(args), "Major compression parameters": compression_config})
    compression_config["advanced_parameters"] = AdvancedCompressionParameters(
        awq_params=AdvancedAWQParameters(prefer_data_aware_scaling=not args.basic_init),
        lora_adapter_rank=args.lora_rank,
    )
    # Configure output and log files.
    output_dir = Path(args.output_dir)
    tensorboard_dir = output_dir / "tb" / datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    last_dir = output_dir / f"last_bits_{args.bits}"
    if not args.resume:
        shutil.rmtree(last_dir, ignore_errors=True)
    for path in [output_dir, tensorboard_dir, last_dir]:
        path.mkdir(exist_ok=True, parents=True)
    ckpt_file = last_dir / "nncf_checkpoint.pth"
    print(f"To visualize the loss and validation metrics, open Tensorboard using the logs from: {tensorboard_dir}")
    tb = SummaryWriter(tensorboard_dir, "QAT with absorbable LoRA")

    # Load original model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(args.pretrained, torch_dtype=torch_dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    # Prepare training and calibration data
    dataset_map = {
        "pile-10k": get_pile_10k,
        "wikitext-2": get_wikitext2,
        "ultrachat_200k": get_ultrachat_200k,
    }
    load_fn = dataset_map[args.dataset]
    train_loader = load_fn(
        num_samples=args.num_train_samples, seqlen=args.train_seqlen, tokenizer=tokenizer, device=device
    )
    if args.basic_init:
        example_input = {k: v.to(device) for k, v in model.dummy_inputs.items()}
        dataset = Dataset([example_input])
    else:
        if args.dataset == "ultrachat_200k":
            calib_loader = train_loader[:129]
        else:
            calib_loader = load_fn(num_samples=129, seqlen=128, tokenizer=tokenizer, device=device)
        dataset = Dataset(map(get_model_input, calib_loader))

    is_ptq = args.scale_epochs == 0 and args.epochs == 0
    orig_hiddens = None
    if not is_ptq:
        # Pre-compute hiddens of teacher model for distillation loss.
        orig_hiddens = calc_hiddens(model, train_loader)

    # Create or load model to tune with Fake Quantizers and absorbable LoRA adapters.
    if args.resume and ckpt_file.exists():
        model = load_checkpoint(model, ckpt_file)
    else:
        if args.equalize_mlp:
            n_eq = equalize_down_proj(model, use_align_scale=True)
            print(f"Equalized {n_eq} down_proj layers.")
            n_eq = equalize_up_gate_with_layernorm(model, use_align_scale=True)
            print(f"Equalized {n_eq} up_proj/gate_proj layers with preceding LayerNorm.")
        model = compress_weights(model, dataset=dataset, **compression_config)
        save_checkpoint(model, ckpt_file, model_state=not args.basic_init)
    fq_lr = args.lr / 10
    weight_decay = args.lr
    param_to_train = set_trainable(model, lora_lr=args.lr, fq_lr=fq_lr)
    opt = torch.optim.AdamW(param_to_train, weight_decay=weight_decay)

    # Run tuning with distillation loss and validation after each epoch.
    grad_accumulation_steps = args.batch_size // args.microbatch_size
    num_samples = len(train_loader)
    epoch_samples = num_samples - num_samples % args.microbatch_size
    microbatches_per_epoch = epoch_samples // args.microbatch_size

    save_model_state = not args.basic_init or args.equalize_mlp
    total_steps = run_training(
        model=model,
        train_loader=train_loader,
        orig_hiddens=orig_hiddens,
        optimizer=opt,
        scheduler=get_linear_lr_scheduler(
            opt, args.linear_lr_scheduler, args.epochs, microbatches_per_epoch, grad_accumulation_steps
        ),
        ckpt_file=ckpt_file,
        tb=tb,
        num_epochs=args.epochs,
        phase_desc="Train epoch",
        start_total_steps=0,
        device=device,
        torch_dtype=torch_dtype,
        grad_accumulation_steps=grad_accumulation_steps,
        num_samples=num_samples,
        epoch_samples=epoch_samples,
        microbatches_per_epoch=microbatches_per_epoch,
        model_state=save_model_state,
    )

    # Optional scales-only finetuning phase: LoRA adapters are frozen, only quantizer scales are trained.
    if args.scale_epochs > 0:
        scale_params = set_trainable(model, lora_lr=args.lr, fq_lr=fq_lr, scales_only=True)
        opt = torch.optim.AdamW(scale_params, weight_decay=weight_decay)
        run_training(
            model=model,
            train_loader=train_loader,
            orig_hiddens=orig_hiddens,
            optimizer=opt,
            scheduler=get_linear_lr_scheduler(
                opt,
                args.linear_lr_scheduler,
                args.scale_epochs,
                microbatches_per_epoch,
                grad_accumulation_steps,
            ),
            ckpt_file=ckpt_file,
            tb=tb,
            num_epochs=args.scale_epochs,
            phase_desc="Scales-only epoch",
            start_total_steps=total_steps,
            device=device,
            torch_dtype=torch_dtype,
            grad_accumulation_steps=grad_accumulation_steps,
            num_samples=num_samples,
            epoch_samples=epoch_samples,
            microbatches_per_epoch=microbatches_per_epoch,
            model_state=save_model_state,
        )

    if is_ptq:
        save_checkpoint(model, ckpt_file, model_state=save_model_state)

    del model

    if args.save_pt:
        export_to_dequantized_torch(args.pretrained, ckpt_file, ckpt_file.parent / "dequantized")
        print(f"The finetuned model has been exported to torch and saved to: {ckpt_file.parent / 'dequantized'}\n")

    # Export the best tuned model to OpenVINO.
    export_to_openvino(args.pretrained, ckpt_file, ckpt_file.parent)


if __name__ == "__main__":
    main(sys.argv[1:])
