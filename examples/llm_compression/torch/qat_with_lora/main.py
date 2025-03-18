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
import random
import shutil
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Union
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
from nncf.quantization.quantize_model import compress_weights
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import BaseWeightsDecompressor
from nncf.torch.quantization.layers import SymmetricLoraQuantizer

MODEL_ID = "HuggingFaceTB/SmolLM-1.7B-Instruct"
DEVICE = "cuda"
TORCH_DTYPE = torch.bfloat16


ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR = ROOT / "output"
TENSORBOARD_DIR = OUTPUT_DIR / "tb"
LAST_DIR = OUTPUT_DIR / "last"
BEST_DIR = LAST_DIR / "best"
for path in [OUTPUT_DIR, TENSORBOARD_DIR, LAST_DIR, BEST_DIR]:
    path.mkdir(exist_ok=True, parents=True)
WWB_REF_FILE = OUTPUT_DIR / "wwb_ref.csv"


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
        print("exit args=", args)
        for decompressor, dtype in self.modules_map.items():
            decompressor.result_dtype = dtype


def get_wikitext2(nsamples, seqlen, tokenizer):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    limit = nsamples * seqlen // 4  # ~1k for 128 samples with seqlen=32 to be aligned with optimum
    text = "".join([" \n" if s == "" else s for s in traindata["text"][:limit]])
    trainenc = tokenizer(text, return_tensors="pt")
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(DEVICE)
        # TODO: or recompute attention_mask/position_ids on tuning?
        attention_mask = torch.ones_like(inp)
        position_ids = torch.cumsum(attention_mask, axis=1) - 1
        trainloader.append({"input_ids": inp, "attention_mask": attention_mask, "position_ids": position_ids})
    return trainloader


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_wwb_ref(model, tokenizer):
    if not WWB_REF_FILE.exists():
        wwb_eval = TextEvaluator(base_model=model, tokenizer=tokenizer, use_chat_template=True)
        wwb_eval.dump_gt(str(WWB_REF_FILE))


def get_similarity(model, wwb_eval, ir_dir):
    print("#" * 50 + " Evaluate via WWB" + "#" * 50)
    model = nncf.strip(model)
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
    _, all_metrics = wwb_eval.score(ov_model)
    return float(all_metrics["similarity"].iloc[0])


def print_trainable_parameters(module):
    params = list(module.parameters())
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    all_param = sum(p.numel() for p in params)
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )


@torch.inference_mode()
def calc_hiddens(model, dataloader):
    orig_hiddens = []
    for i in trange(len(dataloader), total=len(dataloader), desc="Calculating original hiddens", leave=False):
        orig_hiddens.append(model.model(**dataloader[i]).last_hidden_state)
    return orig_hiddens


def kl_div(student_hiddens, teacher_hiddens):
    C = student_hiddens.shape[-1]  # num classes
    return F.kl_div(
        input=F.log_softmax(student_hiddens.view(-1, C), dim=-1),
        target=F.log_softmax(teacher_hiddens.view(-1, C), dim=-1),
        log_target=True,
        reduction="batchmean",
    )


def set_trainable(model, lora_lr, fq_lr):
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
    print_trainable_parameters(model)
    return [{"params": adapters_to_train, "lr": lora_lr}, {"params": scales_to_train, "lr": fq_lr}]


def main():
    assert torch.cuda.is_available()
    set_seed(42)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    save_wwb_ref(model, tokenizer)

    train_loader = get_wikitext2(nsamples=1024, seqlen=1024, tokenizer=tokenizer)
    orig_hiddens = calc_hiddens(model, train_loader)

    example_input = train_loader[0]
    model = compress_weights(
        model,
        mode=CompressWeightsMode.INT4_ASYM,
        group_size=64,
        dataset=Dataset([example_input]),
        compression_format=CompressionFormat.FQ_LORA,
    )

    microbatch_size = 2
    batch_size = 32
    grad_accumulation_steps = batch_size // microbatch_size
    num_samples = len(train_loader)
    epoch_samples = num_samples - num_samples % microbatch_size
    microbatches_per_epoch = epoch_samples // microbatch_size

    tb = SummaryWriter(TENSORBOARD_DIR, "QAT with absorbable LoRA")

    wwb_eval = TextEvaluator(
        tokenizer=tokenizer, gt_data=WWB_REF_FILE, test_data=str(WWB_REF_FILE), use_chat_template=True
    )
    best_similarity = get_similarity(model, wwb_eval, LAST_DIR)
    print(f"WWB similarity for initial 4bit model= {best_similarity:.4f}")
    lm_head = deepcopy(model.lm_head)
    lm_head.requires_grad_(False)

    param_to_train = set_trainable(model, lora_lr=5e-4, fq_lr=5e-5)
    opt = torch.optim.AdamW(param_to_train, weight_decay=5e-4)
    model.train()

    aggregated_loss = float("nan")
    loss_numerator = grad_steps = total_microbatches = 0
    for epoch in range(32):
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        for batch_indices in tqdm(batch_indices_epoch, desc=f"Train epoch {epoch}", leave=[False]):
            batch_indices = batch_indices.tolist()
            total_microbatches += 1

            def form_batch(inputs: List[Union[Dict[str, Tensor], Tensor]], indices: List[int]):
                if isinstance(inputs[0], dict):
                    batch = {name: torch.cat([inputs[i][name] for i in indices], dim=0) for name in inputs[0]}
                else:
                    batch = torch.cat([inputs[i] for i in indices], dim=0).to(device=DEVICE, dtype=TORCH_DTYPE)
                return batch

            inputs = form_batch(train_loader, batch_indices)
            with torch.no_grad():
                targets = lm_head(form_batch(orig_hiddens, batch_indices))
                if hasattr(model.config, "final_logit_softcapping"):  # Gemma
                    fls = model.config.final_logit_softcapping
                    if fls is not None:
                        targets = targets / fls
                        targets = torch.tanh(targets)
                        targets = targets * fls

            outputs = model(**inputs).logits
            loss = kl_div(outputs, targets.to(dtype=TORCH_DTYPE))

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

        smlr = get_similarity(model, wwb_eval, LAST_DIR)
        print(f"WWB similarity = {smlr:.4f}")
        tb.add_scalar("similarity", smlr, total_microbatches)
        if smlr > best_similarity:
            print(f"New best WWB similarity = {smlr:.4f}")
            best_similarity = smlr
            shutil.copytree(LAST_DIR, BEST_DIR, dirs_exist_ok=True)

    print(f"Finetuned OV model has similarity={best_similarity} and is located here: {BEST_DIR}")


if __name__ == "__main__":
    main()
