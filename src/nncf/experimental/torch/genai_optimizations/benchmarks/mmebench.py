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

# This logic is largely copied from the https://github.com/starreeze/efuf/blob/main/evaluate/mme/calculation.py
import os
import string
from argparse import ArgumentParser

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tqdm import tqdm
from transformers import AutoProcessor
from transformers import set_seed

from nncf.experimental.torch.genai_optimizations import get_inputs_embeds
from nncf.experimental.torch.genai_optimizations.benchmarks.utils import add_visual_pruning_args


class MetricCalculator:
    def divide_chunks(self, all_items, n=2):
        for i in range(0, len(all_items), n):
            yield all_items[i : i + n]
        return

    def parse_pred_ans(self, pred_ans):
        pred_ans = pred_ans.lower()
        exclude = set(string.punctuation)
        pred_ans = "".join(ch for ch in pred_ans if ch not in exclude)

        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]
            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"
        return pred_label

    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)
        label_map = {"yes": 1, "no": 0, "other": -1}
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]
        acc = accuracy_score(gts, preds)

        clean_gts, clean_preds = [], []
        other_num = 0
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1, 0])
        precision = precision_score(clean_gts, clean_preds, average="binary")
        recall = recall_score(clean_gts, clean_preds, average="binary")
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        return {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }


def get_model_class(model_name):
    if "Qwen2.5-VL" in model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration
    elif "Qwen2-VL" in model_name:
        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration
    elif "llava-1.5" in model_name:
        from transformers import LlavaForConditionalGeneration

        return LlavaForConditionalGeneration
    elif "llava-v1.6" in model_name:
        from transformers import LlavaNextForConditionalGeneration

        return LlavaNextForConditionalGeneration
    else:
        error_msg = f"Unsupported model class for: {model_name}"
        raise ValueError(error_msg)


def evaluate(args):
    model_name = args.model
    category = args.subset
    dataset = load_dataset("darkyarding/MME", split="test")
    dataset = dataset.filter(lambda x: x["category"] == category)
    metric_util = MetricCalculator()

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model_cls = get_model_class(model_name)
    model = model_cls.from_pretrained(
        model_name,
        trust_remote_code=True,
        # attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN", None),
        temperature=None,
        top_p=None,
        top_k=None,
    ).eval()

    if args.enable_visual_pruning:
        print(f"Enable visual token pruning with num_keep_tokens={args.num_keep_tokens}, theta={args.theta}")
        num_keep_tokens = args.num_keep_tokens
        theta = args.theta
    else:
        num_keep_tokens = None
        theta = None

    all_items = []
    with torch.no_grad():
        for example in tqdm(dataset):
            prompt = example["question"]
            answer = example["answer"].strip().lower()
            image = example["image"].convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}],
                }
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

            image_embeds = get_inputs_embeds(model, inputs, num_keep_tokens=num_keep_tokens, theta=theta)
            kwargs = {}
            if "image_sizes" in inputs:
                kwargs["image_sizes"] = inputs.image_sizes

            generate_ids = model.generate(
                inputs_embeds=image_embeds,
                max_new_tokens=512,
                do_sample=False,
                **kwargs,
            )

            response = processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            pred_label = metric_util.parse_pred_ans(response)
            all_items.append((image, prompt, answer, pred_label))

    grouped = list(metric_util.divide_chunks(all_items, n=2))

    acc_plus_correct = 0
    flat_preds = []
    flat_gts = []

    for pair in grouped:
        if len(pair) != 2:
            continue
        img_correct = 0
        for _, _, gt, pred in pair:
            flat_preds.append(pred)
            flat_gts.append(gt)
            if gt == pred:
                img_correct += 1
        if img_correct == 2:
            acc_plus_correct += 1

    metrics = metric_util.compute_metric(flat_gts, flat_preds)
    acc_plus = acc_plus_correct / len(grouped)
    metrics["acc_plus"] = acc_plus

    print(f"\n MME Evaluation for '{category}'")
    for k, v in metrics.items():
        print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")


if __name__ == "__main__":
    set_seed(42)

    eval_type_dict = [
        "existence",
        "count",
        "position",
        "color",
        "posters",
        "celebrity",
        "scene",
        "landmark",
        "artwork",
        "OCR",
    ] + ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Huggingface model repo")
    parser.add_argument("--subset", choices=eval_type_dict, required=True, help="MME category name")

    add_visual_pruning_args(parser)
    args = parser.parse_args()

    evaluate(args)
