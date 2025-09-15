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
#
# This file includes utility functions copied from the MileBench repository:
# https://github.com/MileBench/MileBench
#
# Licensed under the Apache License

import json
import os
import re
import shutil
from argparse import ArgumentParser

import numpy as np
import requests
import torch
import transformers
from PIL import Image
from rouge import Rouge
from tqdm import tqdm
from transformers import AutoProcessor

from nncf import nncf_logger
from nncf.experimental.torch.genai_optimizations import get_inputs_embeds
from nncf.experimental.torch.genai_optimizations.benchmarks.utils import add_visual_pruning_args


class MileBenchDataset:
    def __init__(self, data_dir, subset, subset_size=200):
        self.data_dir = data_dir
        self.subset = subset
        self.subset_size = subset_size

        self._download_data()
        annotation_path = os.path.join(self.data_dir, self.subset, f"{self.subset}.json")
        with open(annotation_path) as f:
            self.annotation = json.load(f)

        self.image_dir = os.path.join(self.data_dir, self.subset, "images")

    def _download_data(self):
        LINKS = {
            "MileBench_part0.tar.gz": "https://huggingface.co/datasets/FreedomIntelligence/MileBench/resolve/main/MileBench_part0.tar.gz",
            "MileBench_part1.tar.gz": "https://huggingface.co/datasets/FreedomIntelligence/MileBench/resolve/main/MileBench_part1.tar.gz",
            "MileBench_part2.tar.gz": "https://huggingface.co/datasets/FreedomIntelligence/MileBench/resolve/main/MileBench_part2.tar.gz",
            "MileBench_part3.tar.gz": "https://huggingface.co/datasets/FreedomIntelligence/MileBench/resolve/main/MileBench_part3.tar.gz",
            "MileBench_part4.tar.gz": "https://huggingface.co/datasets/FreedomIntelligence/MileBench/resolve/main/MileBench_part4.tar.gz",
            "MileBench_part5.tar.gz": "https://huggingface.co/datasets/FreedomIntelligence/MileBench/resolve/main/MileBench_part5.tar.gz",
        }

        SUBSET2ARCHIVE = {
            # Realistic Temporal
            "ActionLocalization": "MileBench_part0.tar.gz",
            "ActionPrediction": "MileBench_part0.tar.gz",
            "ActionSequence": "MileBench_part0.tar.gz",
            "CharacterOrder": "MileBench_part0.tar.gz",
            "CounterfactualInference": "MileBench_part1.tar.gz",
            "EgocentricNavigation": "MileBench_part1.tar.gz",
            "MovingAttribute": "MileBench_part2.tar.gz",
            "MovingDirection": "MileBench_part2.tar.gz",
            "ObjectExistence": "MileBench_part3.tar.gz",
            "ObjectInteraction": "MileBench_part3.tar.gz",
            "ObjectShuffle": "MileBench_part3.tar.gz",
            "SceneTransition": "MileBench_part3.tar.gz",
            "StateChange": "MileBench_part3.tar.gz",
            # Realistic Semantic
            "ALFRED": "MileBench_part0.tar.gz",
            "CLEVR-Change": "MileBench_part1.tar.gz",
            "DocVQA": "MileBench_part1.tar.gz",
            "IEdit": "MileBench_part2.tar.gz",
            "MMCoQA": "MileBench_part2.tar.gz",
            "MultiModalQA": "MileBench_part2.tar.gz",
            "nuscenes": "MileBench_part3.tar.gz",
            "OCR-VQA": "MileBench_part4.tar.gz",
            "SlideVQA": "MileBench_part4.tar.gz",
            "Spot-the-Diff": "MileBench_part4.tar.gz",
            "TQA": "MileBench_part5.tar.gz",
            "WebQA": "MileBench_part5.tar.gz",
            "WikiVQA": "MileBench_part5.tar.gz",
            # Diagnostic
            "TextNeedleInAHaystack": "MileBench_part5.tar.gz",
            "ImageNeedleInAHaystack": "MileBench_part2.tar.gz",
            "GPR1200": "MileBench_part1.tar.gz",
        }

        archive_name = SUBSET2ARCHIVE.get(self.subset)
        archive_url = LINKS[archive_name]
        archive_path = os.path.join(self.data_dir, archive_name)
        dir_name = os.path.join(self.data_dir, self.subset)

        if not os.path.exists(dir_name):
            if not os.path.exists(archive_path):
                nncf_logger.info(f"Downloading {archive_name} from {archive_url}...")
                os.makedirs(self.data_dir, exist_ok=True)
                response = requests.get(archive_url, stream=True)
                response.raise_for_status()
                with open(archive_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                nncf_logger.info(f"Downloaded archive to {archive_path}")
            else:
                nncf_logger.info(f"Archive already exists at {archive_path}")

            nncf_logger.info(f"Extracting {archive_path}...")
            shutil.unpack_archive(archive_path, self.data_dir)
            nncf_logger.info(f"Extracted to {self.data_dir}")
        else:
            nncf_logger.info(f"Already extracted to {self.data_dir}")

    def __len__(self):
        return min(self.annotation["meta_data"]["num_sample"], self.subset_size)

    @staticmethod
    def _transform_string(s: str) -> str:
        counter = iter(range(1, s.count("{i}") + 1))
        return re.sub(r"\{i\}", lambda _: str(next(counter)), s)

    @staticmethod
    def _preprocess_image(image_path, max_size=512, min_size=32):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        if max(w, h) > max_size:
            scale_factor = max_size / max(w, h)
        elif min(w, h) < min_size:
            scale_factor = min_size / min(w, h)
        else:
            scale_factor = 1.0  # No scaling needed

        new_size = (int(w * scale_factor), int(h * scale_factor))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    def __getitem__(self, idx):
        if idx >= len(self) or idx < 0:
            error_msg = "Index out of range for the dataset."
            raise IndexError(error_msg)

        ann = self.annotation["data"][idx]
        task_instructions = self.annotation["meta_data"]["task_instruction"]

        context = ann["task_instance"]["context"]
        if "choice_list" in ann["task_instance"]:
            choice_str = "\nChoice list: \n"
            choice_str += "\n".join(
                [(f"{chr(65 + idx)}. ") + f"{item}" for idx, item in enumerate(ann["task_instance"]["choice_list"])]
            )
            choice_str += "\nYour answer is: "
            context += choice_str

        img_num = len(ann["task_instance"]["images_path"])
        placeholder = ""
        for i in range(img_num):
            rmv_txt = f"{{image#{i + 1}}}"
            rmv_tbl = f"{{table#{i + 1}}}"
            context = context.replace(rmv_txt, placeholder)
            context = context.replace(rmv_tbl, placeholder)

        task_instruction_id = ann["task_instruction_id"]
        context_str = task_instructions[task_instruction_id] + "\n" + context
        prompt = MileBenchDataset._transform_string(context_str)

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]

        images = []
        for p in ann["task_instance"]["images_path"]:
            img_path = os.path.join(self.image_dir, p)
            image = MileBenchDataset._preprocess_image(img_path)
            images.append(image)
            messages[0]["content"].append({"type": "image", "image": image})

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return {
            "prompt": prompt,
            "images": images,
            "gt_answer": ann["response"],
            "choice_list": ann["task_instance"].get("choice_list", None),
        }


class Eval:
    def __init__(self):
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def char(self, index):
        if index < 26:
            return chr(index + 65)
        elif index < 52:
            return "A" + chr(index + 65 - 26)
        else:
            return "B" + chr(index + 65 - 26 - 26)

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (re.search(self.commaStrip, inText) is not None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(self, answer):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = answer.strip('"')
        answer = answer.strip().lower()
        return answer

    def evaluate_rouge(self, predictions):
        rouge = Rouge()
        acc = []
        for res in predictions:
            gt_ans = self.process(res["gt_answer"])
            pred_ans = self.process(res["pred"])
            assert gt_ans != ""
            if pred_ans == "":
                score = 0
            else:
                score = rouge.get_scores(pred_ans, gt_ans)[0]["rouge-l"]["f"]
            acc.append(score)
        return np.mean(acc)

    def match_choice(self, text, option):
        """Return: A B C D..."""

        def preprocess_option_string(option_string):
            # First, preprocess the option text to normalize it
            processed_option = self.process(option_string)

            # Then, escape any special regex characters in the processed option text
            # List of regex special characters that need to be escaped
            special_chars = [
                "\\",
                ".",
                "^",
                "$",
                "*",
                "+",
                "?",
                "{",
                "}",
                "[",
                "]",
                "|",
                "(",
                ")",
            ]
            # Escape the special characters by prefixing them with a backslash
            for char in special_chars:
                if char in processed_option:
                    processed_option = processed_option.replace(char, "\\" + char)
            # escaped_option = escape_special_chars(processed_option)
            return processed_option

        if text == "":
            return "C"
        try:
            # Maybe start from the head
            # 1. Char+Choice: `A. Blastomycosis`
            option_str = "|".join([preprocess_option_string(f"{k} {v}") for k, v in option.items()])
            option_pattern = rf"({option_str})"
            option_res = re.search(option_pattern, text, re.S)  # NOTE we dont use match_all
            if option_res:
                return (option_res.group(0)[0]).upper()

            # 2. Choice: `Blastomycosis`
            option_str = "|".join([preprocess_option_string(v).replace(" ", "") for k, v in option.items()])
            option_pattern = rf"({option_str})"
            option_res = re.search(option_pattern, text.replace(" ", ""), re.S)  # NOTE we dont use match_all
            if option_res:
                for k, v in option.items():
                    if option_res[0].strip() == preprocess_option_string(v).replace(" ", ""):
                        return k.upper()

            # 3. Char: `A` `AB`
            if len(text) in [1, 2] and text.upper() in option:
                return text.upper()

            # use gpt extract

        except Exception as e:
            print(f"something wrong during match_choice {text}: {e}")
            return text
        return "".join([i.upper() for i in text if i.upper() in option])

    def judge_multi_choice(self, sample):
        gt_ans = sample["gt_answer"]
        pred_ans = sample["pred"]
        choice_list = sample["choice_list"]
        assert gt_ans in choice_list
        # Convert choice_list to a dictionary format expected by match_choice
        option_dict = {self.char(i): choice for i, choice in enumerate(choice_list)}

        # Use match_choice to determine the selected answer from pred_ans
        selected_answer = self.match_choice(pred_ans, option_dict)

        # Check if the selected answer matches the ground truth
        gt_ans_chr = self.char(choice_list.index(sample["gt_answer"]))
        if selected_answer == gt_ans_chr:
            return 1, selected_answer
        else:
            return 0, selected_answer

    def process_sample(self, sample):
        sample["gt_answer"] = self.process(sample["gt_answer"])
        sample["pred"] = self.process(sample["pred"])
        for i in range(len(sample["choice_list"])):
            sample["choice_list"][i] = self.process(sample["choice_list"][i])

    def evaluate_multichoice(self, predictions):
        correct = 0
        for sample in predictions:
            self.process_sample(sample)
            score, extracted_answer = self.judge_multi_choice(sample)
            sample["extracted"] = extracted_answer
            sample["result"] = score
            correct += score
        return correct / len(predictions)

    def evaluate_needle(self, predictions, needle=True):
        correct = 0
        for sample in predictions:
            gt_ans = self.process(sample["gt_answer"])
            pred_ans = self.process(sample["pred"])

            if needle:
                score = 1 if gt_ans in pred_ans.split() else 0
            else:
                score = 1 if gt_ans in pred_ans else 0

            sample["result"] = score
            correct += score
        return correct / len(predictions)

    def evaluate(self, predictions, dataset_name, question_type):
        if "NeedleInAHaystack" in dataset_name or "MMCoQA" in dataset_name:
            return self.evaluate_needle(predictions, needle="NeedleInAHaystack" in dataset_name)
        elif question_type == "open-ended":
            return self.evaluate_rouge(predictions)
        elif question_type == "multi-choice":
            return self.evaluate_multichoice(predictions)
        else:
            error_msg = "Dataset not supported"
            raise ValueError(error_msg)


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
    elif "Phi" in model_name:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM
    else:
        error_msg = f"{model_name} is not supported."
        raise NotImplementedError(error_msg)


def evaluate(dataset, processor, model, num_keep_tokens, theta):
    with torch.no_grad():
        answers = []
        for data_sample in tqdm(dataset):
            prompt = data_sample["prompt"]
            images = data_sample["images"]
            inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)

            image_embeds = get_inputs_embeds(model, inputs, num_keep_tokens=num_keep_tokens, theta=theta)
            kwargs = {}
            if "image_sizes" in inputs:
                kwargs["image_sizes"] = inputs.image_sizes

            generate_ids = model.generate(
                inputs_embeds=image_embeds,
                max_new_tokens=64,
                do_sample=False,
                **kwargs,
            )
            response = processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            answers.append(
                {
                    "gt_answer": data_sample["gt_answer"],
                    "choice_list": data_sample["choice_list"],
                    "pred": response,
                }
            )

    question_type = dataset.annotation["meta_data"]["question_type"]
    scorer = Eval()
    score = scorer.evaluate(answers, args.subset, question_type)
    print(f"Score: {score}")


if __name__ == "__main__":
    transformers.set_seed(42)

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--subset", type=str, required=True, help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="MileBench", help="Data directory")

    add_visual_pruning_args(parser)
    args = parser.parse_args()

    dataset = MileBenchDataset(data_dir=args.data_dir, subset=args.subset)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model_cls = get_model_class(args.model)
    model = model_cls.from_pretrained(
        args.model,
        # attn_implementation="eager",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN", None),
        temperature=None,
        top_p=None,
        top_k=None,
    )
    model = model.eval()

    if args.enable_visual_pruning:
        print(f"Enable visual token pruning with num_keep_tokens={args.num_keep_tokens}, theta={args.theta}")
        num_keep_tokens = args.num_keep_tokens
        theta = args.theta
    else:
        num_keep_tokens = None
        theta = None
    evaluate(dataset, processor, model, num_keep_tokens=num_keep_tokens, theta=theta)
