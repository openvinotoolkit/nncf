# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchvision
from datasets import load_dataset
from memory_profiler import memory_usage
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoModelForCausalLM

import nncf
import nncf.experimental
import nncf.experimental.torch
import nncf.experimental.torch.sparsify_activations
from nncf.experimental.torch.sparsify_activations.torch_backend import SparsifyActivationsAlgoBackend
from tests.post_training.pipelines.base import PT_BACKENDS
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.image_classification_timm import ImageClassificationTimm
from tests.post_training.pipelines.lm_weight_compression import LMWeightCompression
from tests.post_training.pipelines.lm_weight_compression import WCTimeStats
from tests.torch.helpers import set_torch_seed


@dataclass
class SparsifyActivationsTimeStats(WCTimeStats):
    """
    Contains statistics that are parsed from the stdout of Sparsify Activations tests.
    """

    time_sparsifier_calibration: Optional[str] = None
    STAT_NAMES = [*WCTimeStats.STAT_NAMES, "Activations Sparsifier calibration time"]
    VAR_NAMES = [*WCTimeStats.VAR_NAMES, "time_sparsifier_calibration"]
    REGEX_PREFIX = [*WCTimeStats.REGEX_PREFIX, SparsifyActivationsAlgoBackend.CALIBRATION_TRACKING_DESC]


class LMSparsifyActivations(LMWeightCompression):
    def prepare_model(self) -> None:
        is_stateful = self.params.get("is_stateful", False)

        if self.backend == BackendType.TORCH:
            if is_stateful:
                raise RuntimeError(f"is_stateful={is_stateful} is not supported for PyTorch backend.")

            self.model_hf = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                attn_implementation="eager",
            )
            self.model = self.model_hf
        elif self.backend in [BackendType.OV, BackendType.FP32]:
            if is_stateful:
                self.fp32_model_dir = self.fp32_model_dir.parent / (self.fp32_model_dir.name + "_sf")
            if not (self.fp32_model_dir / self.OV_MODEL_NAME).exists():
                # export by model_id
                self.model_hf = OVModelForCausalLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    export=True,
                    load_in_8bit=False,
                    compile=False,
                    stateful=is_stateful,
                )
            else:
                # no export, load from IR. Applicable for sequential run of test cases in local environment.
                self.model_hf = OVModelForCausalLM.from_pretrained(
                    self.fp32_model_dir, load_in_8bit=False, compile=False, stateful=is_stateful
                )
            self.model = self.model_hf.model
        else:
            raise RuntimeError(f"backend={self.backend.value} is not supported.")

        if not (self.fp32_model_dir / self.OV_MODEL_NAME).exists():
            self._dump_model_fp32()

    def prepare_calibration_dataset(self):
        dataset = load_dataset("wikitext", "wikitext-2-v1", split="train", revision="b08601e")
        dataset = dataset.filter(lambda example: len(example["text"].split()) > 256)
        subset_size = self.compression_params.get("subset_size") or 64
        dataset = dataset.select(range(subset_size))
        self.calibration_dataset = nncf.Dataset(dataset, partial(self.get_transform_calibration_fn(), max_tokens=256))

    def compress(self) -> None:
        if self.backend == BackendType.FP32:
            return
        start_time = time.perf_counter()
        self.run_info.compression_memory_usage = memory_usage(self._compress, max_usage=True)
        self.run_info.time_compression = time.perf_counter() - start_time

    def collect_data_from_stdout(self, stdout: str):
        stats = SparsifyActivationsTimeStats()
        stats.fill(stdout)
        self.run_info.stats_from_output = stats

    @set_torch_seed(seed=42)
    def _compress(self):
        """
        Actual call of weight compression and/or activation sparsification.
        """
        self.compressed_model = self.model
        if self.compression_params.get("compress_weights", None) is not None:
            self.compressed_model = nncf.compress_weights(
                self.compressed_model,
                dataset=self.calibration_dataset,
                **self.compression_params["compress_weights"],
            )
        if self.compression_params.get("sparsify_activations", None) is not None:
            self.compressed_model = nncf.experimental.torch.sparsify_activations.sparsify_activations(
                self.compressed_model,
                dataset=self.calibration_dataset,
                **self.compression_params["sparsify_activations"],
            )


class ImageClassificationTimmSparsifyActivations(ImageClassificationTimm):
    def compress(self) -> None:
        """
        Run compression of the model and collect time and memory usage information.
        """
        if self.backend == BackendType.FP32:
            # To validate not compressed model
            self.path_compressed_ir = self.fp32_model_dir / "model_fp32.xml"
            return

        if self.backend in PT_BACKENDS:
            inference_num_threads = os.environ.get("INFERENCE_NUM_THREADS")
            if inference_num_threads is not None:
                torch.set_num_threads(int(inference_num_threads))
        else:
            raise RuntimeError(f"backend={self.backend.value} is not supported.")

        start_time = time.perf_counter()
        self.run_info.compression_memory_usage = memory_usage(self._compress, max_usage=True)
        self.run_info.time_compression = time.perf_counter() - start_time

    def collect_data_from_stdout(self, stdout: str):
        stats = SparsifyActivationsTimeStats()
        stats.fill(stdout)
        self.run_info.stats_from_output = stats

    @set_torch_seed(seed=42)
    def _compress(self):
        """
        Actual call of activation sparsification.
        """
        self.compressed_model = self.model
        if self.compression_params.get("sparsify_activations", None) is not None:
            self.compressed_model = nncf.experimental.torch.sparsify_activations.sparsify_activations(
                self.compressed_model,
                dataset=self.calibration_dataset,
                **self.compression_params["sparsify_activations"],
            )

    def prepare_calibration_dataset(self):
        subset_size = self.compression_params.get("subset_size") or 512
        val_dataset = torchvision.datasets.ImageFolder(
            root=self.data_dir / "imagenet" / "val", transform=self.transform
        )
        indices = np.random.default_rng(42).choice(len(val_dataset), size=subset_size, replace=False)
        subset = torch.utils.data.Subset(val_dataset, indices=indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=self.batch_size, num_workers=2, shuffle=False)
        self.calibration_dataset = nncf.Dataset(loader, self.get_transform_calibration_fn())
