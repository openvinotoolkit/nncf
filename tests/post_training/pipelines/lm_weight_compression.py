# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime as dt
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import openvino as ov
from datasets import load_dataset
from memory_profiler import memory_usage
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer
from whowhatbench import Evaluator

import nncf
from tests.post_training.pipelines.base import OV_BACKENDS
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.base import BaseTestPipeline
from tests.post_training.pipelines.base import StatsFromOutput

OV_MODEL_NAME = "openvino_model.xml"


@dataclass
class WCTimeStats(StatsFromOutput):
    time_stat_collection: Optional[str] = None
    time_mixed_precision: Optional[str] = None
    time_awq: Optional[str] = None
    time_apply_compression: Optional[str] = None

    def fill(self, stdout: str):
        """
        Parsing stdout of the test and collect additional data:
         - time of statistic collection
         - time of bias correction
         - time of validation

        :param stdout: stdout text
        """
        mapping = {
            "time_stat_collection": "Statistics collection",
            "time_mixed_precision": "Searching for Mixed-Precision Configuration",
            "time_awq": "Applying AWQ",
            "time_apply_compression": "Applying Weight Compression",
        }
        time_regex = ".*•\s(.*)\s•.*"
        for line in stdout.splitlines():
            for attr_name, prefix_regex in mapping.items():
                match = re.search(r"{}{}".format(prefix_regex, time_regex), line)
                if match:
                    parsed_time = dt.datetime.strptime(match.group(1), "%H:%M:%S")
                    setattr(self, attr_name, parsed_time)
                continue

    def get_result_dict(self):
        return {
            "Stat. collection time": self.time_stat_collection,
            "Mixed-Precision search time": self.time_mixed_precision,
            "AWQ time": self.time_awq,
            "Apply Compression time": self.time_apply_compression,
        }


class LMWeightCompression(BaseTestPipeline):
    """Pipeline for casual language models from Hugging Face repository"""

    def prepare_model(self) -> None:
        if self.backend in OV_BACKENDS + [BackendType.FP32]:
            self.model_hf = OVModelForCausalLM.from_pretrained(
                self.model_id, export=True, load_in_8bit=False, compile=False, stateful=False
            )
            self.model = self.model_hf.model
        self._dump_model_fp32()

    def _dump_model_fp32(self) -> None:
        """Dump IRs of fp32 models, to help debugging."""
        if self.backend in OV_BACKENDS + [BackendType.FP32]:
            self.model_hf.save_pretrained(self.output_model_dir)
            self.model_hf._save_config(self.output_model_dir)

    def prepare_preprocessor(self) -> None:
        self.preprocessor = AutoTokenizer.from_pretrained(self.model_id)

    def get_transform_calibration_fn(self):
        if self.backend in OV_BACKENDS:

            def transform_fn(data):
                tokenized_text = self.preprocessor(data["text"], return_tensors="np")
                input_ids = tokenized_text["input_ids"]
                attention_mask = tokenized_text["attention_mask"]

                inputs = {}
                inputs["input_ids"] = input_ids
                inputs["attention_mask"] = tokenized_text["attention_mask"]
                position_ids = np.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1

                # The magic forms KV cache as model inputs
                batch_size = input_ids.shape[0]
                for input_name in self.model_hf.key_value_input_names:
                    model_inputs = self.model.input(input_name)
                    shape = model_inputs.get_partial_shape()
                    shape[0] = batch_size
                    if shape[2].is_dynamic:
                        shape[2] = 0
                    else:
                        shape[1] = 0
                    inputs[input_name] = ov.Tensor(model_inputs.get_element_type(), shape.get_shape())

                inputs["position_ids"] = position_ids
                return inputs

        return transform_fn

    def prepare_calibration_dataset(self):
        dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
        dataset = dataset.filter(lambda example: len(example["text"]) > 80)
        self.calibration_dataset = nncf.Dataset(dataset, self.get_transform_calibration_fn())

    def _quantize(self):
        """
        Quantize self.model
        """
        self._compressed_model_dir = self.output_model_dir / "compressed"
        self.quantized_model = nncf.compress_weights(
            self.model,
            dataset=self.calibration_dataset,
            **self.compression_params,
        )
        self._compressed_model_dir.mkdir(parents=True, exist_ok=True)
        ov.serialize(self.model, self._compressed_model_dir / OV_MODEL_NAME)
        self.model_hf._save_config(self._compressed_model_dir)

    def cleanup_cache(self):
        dir_with_cache = 'model_cache'
        shutil.rmtree(path)

        self._compressed_model_dir / dir_with_cache
        self.output_model_dir / dir_with_cache

        # TODO: general cleanup, remove model_cache for OV as well
        pass

    def quantize(self) -> None:
        """
        Run quantization of the model and collect time and memory usage information.
        """
        if self.backend == BackendType.FP32:
            # To validate not quantized model
            self.path_quantized_ir = self.output_model_dir / OV_MODEL_NAME
            return

        print("Weight compression...")
        start_time = time.perf_counter()
        self.run_info.quant_memory_usage = memory_usage(self._quantize, max_usage=True)
        self.run_info.time_quantization = time.perf_counter() - start_time

    def get_num_quantized(self) -> None:
        # TODO: 0 FQ, but number of weights compressed
        # TODO: maybe should be removed from base!
        # can calculate u4 or u8 constants? or by certain name? 'fq'
        pass

    def _validate(self):
        core = ov.Core()

        if os.environ.get("CPU_THREADS_NUM"):
            # Set CPU_THREADS_NUM for OpenVINO inference
            cpu_threads_num = os.environ.get("CPU_THREADS_NUM")
            core.set_property("CPU", properties={"CPU_THREADS_NUM": str(cpu_threads_num)})

        gold_folder = self.output_model_dir  # TODO: should be a parameter
        gold_csv = gold_folder / "gold_all.csv"
        print("gold path:", gold_csv.resolve())
        if gold_csv.exists():
            evaluator = Evaluator(
                tokenizer=self.preprocessor, gt_data=gold_csv, test_data=str(gold_csv), metrics=("similarity",)
            )
        else:
            model_gold = OVModelForCausalLM.from_pretrained(
                gold_folder, trust_remote_code=True, load_in_8bit=False, compile=False, stateful=False
            )
            evaluator = Evaluator(base_model=model_gold, tokenizer=self.preprocessor, metrics=("similarity",))
            evaluator.dump_gt(str(gold_csv))

        compressed_model_hf = OVModelForCausalLM.from_pretrained(
            self._compressed_model_dir, trust_remote_code=True, load_in_8bit=False, compile=False, stateful=False
        )

        _, all_metrics = evaluator.score(compressed_model_hf)
        similarity = all_metrics["similarity"][0]
        self.run_info.metric_name = "Similarity"
        self.run_info.metric_value = similarity

    def collect_data_from_stdout(self, stdout: str):
        self.run_info.stats_from_output = WCTimeStats(stdout)

    @staticmethod
    def cleanup_cache():
        """
        Helper for removing cached model representation.

        """
        # TODO: remove model_cache

    def save_quantized_model(self) -> None:
        """
        Save quantized model to IR.
        """

    def run_bench(self) -> None:
        """
        Run a benchmark to collect performance statistics.
        """
