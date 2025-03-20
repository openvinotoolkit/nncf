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
import random
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from weakref import WeakKeyDictionary

import torch
import torch.nn.functional as F
from datasets import load_dataset
from optimum.exporters.openvino.convert import export_from_model
from optimum.intel.openvino import OVModelForCausalLM
from torch import Tensor
from torch import nn
from torch.jit import TracerWarning
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm import trange
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from whowhatbench import TextEvaluator

import nncf
from nncf.data.dataset import Dataset
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.parameters import StripFormat
from nncf.quantization.quantize_model import compress_weights
from nncf.torch.model_creation import load_from_config
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import BaseWeightsDecompressor
from nncf.torch.quantization.layers import SymmetricLoraQuantizer


# TODO: (nlyalyus) move to Optimum-Intel (ticket 164159)
class PatchDecompressorDtype:
    """
    Patching of compression modules in order to export bfloat16 models to OV.
    """

    def __init__(self, model):
        self.model = model
        self.modules_map: WeakKeyDictionary[nn.Module, List[str]] = WeakKeyDictionary()

    def __enter__(self):
        model_layout = self.model.nncf.transformation_layout()
        transformations = model_layout.transformations
        for command in transformations:
            decompressor = command.fn
            if isinstance(decompressor, BaseWeightsDecompressor):
                self.modules_map[decompressor] = decompressor.result_dtype
                decompressor.result_dtype = torch.float32

    def __exit__(self, *args):
        for decompressor, dtype in self.modules_map.items():
            decompressor.result_dtype = dtype


def get_wikitext2(nsamples: int, seqlen: int, tokenizer: Any, device: torch.device) -> List[Tensor]:
    """
    Loads and processes the Wikitext-2 dataset for training.

    :param nsamples: Number of samples to generate.
    :param seqlen: Sequence length for each sample.
    :param tokenizer: Tokenizer to encode the text.
    :param device: Device to move the tensors to (e.g., 'cpu' or 'cuda').
    :return: A list of tensors containing the tokenized text samples.
    """
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    limit = nsamples * seqlen // 4  # ~1k for 128 samples with seqlen=32 to be aligned with optimum
    text = "".join([" \n" if s == "" else s for s in traindata["text"][:limit]])
    trainenc = tokenizer(text, return_tensors="pt")
    trainloader = []
    for _ in range(nsamples):
        i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        trainloader.append(inp)
    return trainloader


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.inference_mode()
def save_wwb_ref(model: torch.nn.Module, tokenizer: Any, wwb_ref_file: Path, device: torch.device) -> None:
    """
    Save the reference answers for the WWB (WhoWhatBenchmark) evaluation.

    :param model: The model to be evaluated.
    :param tokenizer: The tokenizer used for processing text inputs.
    :param wwb_ref_file: The file path where the reference answers will be saved.
    :param device: The device to which the model should be moved after evaluation.
    """

    if not wwb_ref_file.exists():
        print("#" * 50 + " Collect reference answers for WWB " + "#" * 50)
        model = model.to("cpu")  # TODO: (nlyalyus) remove when WWB will be fixed for cuda.
        wwb_eval = TextEvaluator(base_model=model, tokenizer=tokenizer, use_chat_template=True)
        wwb_eval.dump_gt(str(wwb_ref_file))
        model = model.to(device)
        torch.cuda.empty_cache()


@torch.inference_mode()
def measure_similarity(model: nn.Module, tokenizer: Any, wwb_ref_file: Path, ir_dir: Path) -> float:
    """
    Measures the similarity of a model's output to a reference outputs from a given file using WWB evaluation.

    :param model: The model to be evaluated.
    :param tokenizer: The tokenizer used for processing text data.
    :param wwb_ref_file: The file path to the reference data for WWB evaluation.
    :param ir_dir: The directory where the intermediate representation (IR) of the model will be stored.
    :return: The similarity score as a float.
    """

    print("#" * 50 + " Evaluate via WWB " + "#" * 50)
    model = nncf.strip(model, strip_format=StripFormat.DQ)
    with PatchDecompressorDtype(model), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=TracerWarning)
        export_from_model(model.cpu(), ir_dir, patch_16bit_model=True, device="cpu")
        ov_model = OVModelForCausalLM.from_pretrained(
            model_id=ir_dir,
            trust_remote_code=True,
            load_in_8bit=False,
            compile=True,
            ov_config={"KV_CACHE_PRECISION": "f16", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"},
        )
    wwb_eval = TextEvaluator(
        tokenizer=tokenizer, gt_data=wwb_ref_file, test_data=str(wwb_ref_file), use_chat_template=True
    )
    _, all_metrics = wwb_eval.score(ov_model)
    torch.cuda.empty_cache()
    return float(all_metrics["similarity"].iloc[0])


@torch.inference_mode()
def calc_hiddens(model: nn.Module, dataloader: List[Tensor]):
    """
    Calculate the hidden states for each input in the dataloader using the given model.

    :param model: The model used to calculate the hidden states.
    :param dataloader: The dataloader providing the inputs to the model.
    :return: A list of hidden states for each input in the dataloader.
    """
    orig_hiddens = []
    for i in trange(len(dataloader), total=len(dataloader), desc="Calculating original hiddens", leave=False):
        model_input = get_model_input(dataloader[i])
        orig_hiddens.append(model.model(**model_input).last_hidden_state)
    torch.cuda.empty_cache()
    return orig_hiddens


def get_model_input(input_ids: Tensor) -> Dict[str, Tensor]:
    """
    Prepares the model input dictionary with input IDs, attention mask, and position IDs.

    :param input_ids: Tensor containing the input IDs.
    :return: A dictionary with keys "input_ids", "attention_mask", and "position_ids",
        each mapping to their respective tensors.
    """
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.cumsum(attention_mask, axis=1) - 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}


def kl_div(student_hiddens: torch.Tensor, teacher_hiddens: torch.Tensor):
    """
    Computes the Kullback-Leibler divergence loss between the student and teacher hidden states.
    The input tensors are expected to have the same shape, and the last dimension represents the number of classes.

    :param student_hiddens: The hidden states from the student model.
    :param teacher_hiddens: The hidden states from the teacher model.
    :returns: The computed KL divergence loss.
    """
    C = student_hiddens.shape[-1]  # num classes
    return F.kl_div(
        input=F.log_softmax(student_hiddens.view(-1, C), dim=-1),
        target=F.log_softmax(teacher_hiddens.view(-1, C), dim=-1),
        log_target=True,
        reduction="batchmean",
    )


def set_trainable(model: nn.Module, lora_lr: float, fq_lr: float) -> List[Dict[str, Any]]:
    """
    Sets the trainable parameters of the model for quantization-aware training with LoRA (Low-Rank Adaptation).

    This function disables gradients for all parameters in the model, then selectively enables gradients for
    specific quantizers (AsymmetricLoraQuantizer, SymmetricLoraQuantizer) that have 4-bit quantization.
    It collects the trainable parameters and adapters from these quantizers and returns them in a format
    suitable for an optimizer.

    :param model: The model to be trained.
    :param lora_lr:  Learning rate for the LoRA adapters.
    :param fq_lr:  Learning rate for the quantizer scales.
    :returns : A list of dictionaries containing the parameters to be optimized and their corresponding learning rates.
    """
    model.requires_grad_(False)
    scales_to_train = []
    adapters_to_train = []
    transformations = model.nncf.transformation_layout().transformations
    for command in transformations:
        quantizer = command.fn
        if isinstance(quantizer, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer)) and (quantizer.num_bits == 4):
            quantizer.enable_gradients()
            params = quantizer.get_trainable_params()
            adapters = quantizer.get_adapters()
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
    return [{"params": adapters_to_train, "lr": lora_lr}, {"params": scales_to_train, "lr": fq_lr}]


def save_checkpoint(model: nn.Module, ckpt_file: Path) -> None:
    """
    Saves the state of a tuned model from a checkpoint.

    :param model: The model to load the checkpoint into.
    :param example_input: An example input that will be used for model tracing. It's required to insert and run FQs.
    :param ckpt_file: Path to the checkpoint file.
    """

    ckpt = {"nncf_state_dict": model.nncf.state_dict(), "nncf_config": model.nncf.get_config()}
    torch.save(ckpt, ckpt_file)


def load_checkpoint(model: nn.Module, example_input: Any, ckpt_file: Path):
    """
    Loads the state of a tuned model from a checkpoint. This function restores the placement of Fake Quantizers (FQs)
    with absorbable LoRA adapters and loads their parameters.

    :param model: The model to load the checkpoint into.
    :param example_input: An example input that will be used for model tracing. It's required to insert and run FQs.
    :param ckpt_file: Path to the checkpoint file.
    :returns: The model with the loaded NNCF state from checkpoint.
    """
    ckpt = torch.load(ckpt_file, weights_only=False)
    model = load_from_config(model, ckpt["nncf_config"], example_input=example_input)
    model.nncf.load_state_dict(ckpt["nncf_state_dict"])
    return model


def get_argument_parser():
    parser = argparse.ArgumentParser(add_help=True)

    # Model params
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="HuggingFaceTB/SmolLM-1.7B-Instruct",
        help="The model id or path of a pretrained HF model configuration.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to the directory for storing logs, tuning checkpoint, compressed model, validation references.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to start from previously saved checkpoint. If not specified or checkpoint does not exist, "
        "start from scratch by post-training weight compression initializion.",
    )

    # Data params
    parser.add_argument("--nsamples", type=int, default=1024, help="Number of training samples")
    parser.add_argument("--seqlen", type=int, default=512, help="Calibration data context length.")

    # Training params
    parser.add_argument("--lr", type=float, default=1e-4, help="Finetuning learning rate.")
    parser.add_argument("--epochs", type=int, default=32, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Size of training batch.")
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=2,
        help="Size of each training microbatch. Gradients will be accumulated until the batch size is reached.",
    )
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    assert torch.cuda.is_available()
    set_seed(42)
    device = "cuda"
    torch_dtype = torch.bfloat16
    compression_config = dict(
        mode=CompressWeightsMode.INT4_ASYM, group_size=64, compression_format=CompressionFormat.FQ_LORA
    )

    # Configure output and log files.
    output_dir = Path(args.output_dir)
    tensorboard_dir = output_dir / "tb" / datetime.now().strftime("%D-%T")
    last_dir = output_dir / "last"
    best_dir = output_dir / "best"
    for path in [output_dir, tensorboard_dir, last_dir, best_dir]:
        if not args.resume:
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(exist_ok=True, parents=True)
    wwb_ref_file = output_dir / "wwb_ref.csv"
    ckpt_file = last_dir / "nncf_checkpoint.pth"
    print(f"To visualize the loss and validation metrics, open Tensorboard using the logs from: {tensorboard_dir}")
    tb = SummaryWriter(tensorboard_dir, "QAT with absorbable LoRA")

    # Load original model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model, torch_dtype=torch_dtype, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    # Use WhoWhatBench tool is for validation during tuning. It estimates the similarity score between embedding
    # computed by for data generated by two models, original floating-point one and optimized.
    save_wwb_ref(model, tokenizer, wwb_ref_file, device)

    # Prepare training data and pre-compute hiddens of teacher model for distillation loss.
    train_loader = get_wikitext2(nsamples=args.nsamples, seqlen=args.seqlen, tokenizer=tokenizer, device=device)
    orig_hiddens = calc_hiddens(model, train_loader)

    # Create or load model to tune with Fake Quantizers and absorbable LoRA adapters.
    example_input = get_model_input(train_loader[0])
    if args.resume and ckpt_file.exists():
        model = load_checkpoint(model, example_input, ckpt_file)
    else:
        model = compress_weights(model, dataset=Dataset([example_input]), **compression_config)
        save_checkpoint(model, ckpt_file)
    fq_lr = args.lr / 10
    weight_decay = args.lr
    param_to_train = set_trainable(model, lora_lr=args.lr, fq_lr=fq_lr)
    opt = torch.optim.AdamW(param_to_train, weight_decay=weight_decay)
    model.train()

    best_similarity = 0  # measure_similarity(model, tokenizer, wwb_ref_file, last_dir)
    print(f"Initial WWB similarity= {best_similarity:.4f}")

    # Run tuning with distillation loss and validation on WWB after each epoch.
    grad_accumulation_steps = args.batch_size // args.microbatch_size
    num_samples = len(train_loader)
    epoch_samples = num_samples - num_samples % args.microbatch_size
    microbatches_per_epoch = epoch_samples // args.microbatch_size
    aggregated_loss = float("nan")
    loss_numerator = grad_steps = total_microbatches = 0
    for epoch in range(args.epochs):
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        for indices in tqdm(batch_indices_epoch, desc=f"Train epoch {epoch}", leave=[False]):
            indices = indices.tolist()
            total_microbatches += 1

            def form_batch(inputs: List[Tensor], model_input: bool):
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
            loss = kl_div(outputs, targets.to(dtype=torch_dtype))

            # Perform an optimization step after accumulating gradients over multiple minibatches.
            loss_numerator += loss.item()
            grad_steps += 1
            if not torch.isfinite(loss).item():
                err = f"Fine-tuning loss is {loss}"
                raise ValueError(err)
            (loss / grad_accumulation_steps).backward()
            if grad_steps == grad_accumulation_steps:
                opt.step()
                opt.zero_grad()
                aggregated_loss = loss_numerator / grad_steps
                loss_numerator = grad_steps = 0
            tb.add_scalar("loss", aggregated_loss, total_microbatches)

        # Export tuned model to OpenVINO and evaluate it using WWB.
        # Save the best checkpoint and OpenVINO IR for the highest similarity score obtained from WWB.
        save_checkpoint(model, ckpt_file)
        smlr = measure_similarity(model, tokenizer, wwb_ref_file, last_dir)
        print(f"[Epoch {epoch}], WWB similarity = {smlr:.4f}")
        tb.add_scalar("similarity", smlr, total_microbatches)
        if smlr > best_similarity:
            print(f"New best WWB similarity = {smlr:.4f}")
            best_similarity = smlr
            shutil.copytree(last_dir, best_dir, dirs_exist_ok=True)

    print(f"The finetuned OV model with the best similarity={best_similarity} saved to: {best_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
