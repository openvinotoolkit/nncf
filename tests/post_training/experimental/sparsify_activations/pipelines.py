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


from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import openvino as ov
import torch
import torch.utils
import torch.utils.data
import torchvision
from datasets import load_dataset
from optimum.exporters.openvino.convert import export_from_model
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoModelForCausalLM

import nncf
from nncf.experimental.torch.sparsify_activations import sparsify_activations
from nncf.experimental.torch.sparsify_activations.sparsify_activations_impl import SparsifyActivationsAlgoBackend
from nncf.experimental.torch.sparsify_activations.torch_backend import PTSparsifyActivationsAlgoBackend
from nncf.torch.quantization.layers import INT8AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8SymmetricWeightsDecompressor
from tests.post_training.pipelines.base import PT_BACKENDS
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.base import BaseTestPipeline
from tests.post_training.pipelines.base import ErrorReason
from tests.post_training.pipelines.base import ErrorReport
from tests.post_training.pipelines.base import PTQNumCompressNodes
from tests.post_training.pipelines.base import RunInfo
from tests.post_training.pipelines.base import get_num_fq_int4_int8
from tests.post_training.pipelines.image_classification_timm import ImageClassificationTimm
from tests.post_training.pipelines.lm_weight_compression import LMWeightCompression
from tests.post_training.pipelines.lm_weight_compression import WCNumCompressNodes
from tests.post_training.pipelines.lm_weight_compression import WCTimeStats
from tests.post_training.pipelines.lm_weight_compression import collect_int4_int8_num_errors
from tests.torch.experimental.sparsify_activations.helpers import count_sparsifier_patterns_in_ov
from tests.torch.helpers import set_torch_seed


@dataclass
class SATimeStats(WCTimeStats):
    """
    Contains statistics that are parsed from the stdout of Sparsify Activations tests.
    """

    time_sparsifier_calibration: Optional[str] = None
    STAT_NAMES = [*WCTimeStats.STAT_NAMES, "Activations Sparsifier calibration time"]
    VAR_NAMES = [*WCTimeStats.VAR_NAMES, "time_sparsifier_calibration"]
    REGEX_PREFIX = [*WCTimeStats.REGEX_PREFIX, SparsifyActivationsAlgoBackend.CALIBRATION_TRACKING_DESC]


@dataclass
class SANumCompressNodes(PTQNumCompressNodes, WCNumCompressNodes):
    num_sparse_activations: Optional[int] = None

    def get_data(self):
        data = super().get_data()
        data["Num sparse activations"] = self.num_sparse_activations
        return data


class SAPipelineMixin(BaseTestPipeline):
    """
    Common methods in the test pipeline for Sparsify Activations.
    """

    def __init__(
        self,
        reported_name: str,
        model_id: str,
        backend: BackendType,
        compression_params: dict,
        output_dir: Path,
        data_dir: Path,
        reference_data: dict,
        no_eval: bool,
        run_benchmark_app: bool,
        torch_compile_validation: bool = False,
        params: dict = None,
        batch_size: int = 1,
        memory_monitor: bool = False,
    ):
        super().__init__(
            reported_name,
            model_id,
            backend,
            compression_params,
            output_dir,
            data_dir,
            reference_data,
            no_eval,
            run_benchmark_app,
            torch_compile_validation,
            params,
            batch_size,
            memory_monitor,
        )
        self.run_info = RunInfo(model=reported_name, backend=self.backend, num_compress_nodes=SANumCompressNodes())

    def collect_data_from_stdout(self, stdout: str):
        stats = SATimeStats()
        stats.fill(stdout)
        self.run_info.stats_from_output = stats

    @set_torch_seed(seed=42)
    @torch.no_grad()
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
            self.compressed_model = sparsify_activations(
                self.compressed_model,
                dataset=self.calibration_dataset,
                **self.compression_params["sparsify_activations"],
            )

    def get_num_compressed(self) -> None:
        ie = ov.Core()
        model = ie.read_model(model=self.path_compressed_ir)
        num_fq, num_int4, num_int8 = get_num_fq_int4_int8(model)
        num_sparse_activations = count_sparsifier_patterns_in_ov(model)
        self.run_info.num_compress_nodes.num_fq_nodes = num_fq
        self.run_info.num_compress_nodes.num_int8 = num_int8
        self.run_info.num_compress_nodes.num_int4 = num_int4
        self.run_info.num_compress_nodes.num_sparse_activations = num_sparse_activations

    def collect_errors(self) -> List[ErrorReport]:
        errors = super().collect_errors()
        run_info = self.run_info
        reference_data = self.reference_data
        errors.extend(collect_int4_int8_num_errors(self.run_info, self.reference_data))

        ref_num_sparse_activations = reference_data.get("num_sparse_activations")
        num_sparse_activations = run_info.num_compress_nodes.num_sparse_activations

        if ref_num_sparse_activations is not None and num_sparse_activations != ref_num_sparse_activations:
            status_msg = (
                f"Regression: The number of sparse activations is {num_sparse_activations}, "
                f"which differs from reference {ref_num_sparse_activations}."
            )
            errors.append(ErrorReport(ErrorReason.NUM_COMPRESSED, status_msg))

        return errors


class LMSparsifyActivations(SAPipelineMixin, LMWeightCompression):
    DEFAULT_SUBSET_SIZE = 32

    def prepare_model(self):
        is_stateful = self.params.get("is_stateful", False)

        if self.backend in PT_BACKENDS:
            if is_stateful:
                msg = f"is_stateful={is_stateful} is not supported for PyTorch backend."
                raise RuntimeError(msg)

            self.model_hf = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                device_map="cuda" if self.backend == BackendType.CUDA_TORCH else "cpu",
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
            msg = f"backend={self.backend.value} is not supported."
            raise RuntimeError(msg)

        if not (self.fp32_model_dir / self.OV_MODEL_NAME).exists():
            self._dump_model_fp32()

        # Use FP16 for CUDA_TORCH backend as it is more common when running LLM on CUDA.
        if self.backend == BackendType.CUDA_TORCH:
            self.model_hf.half()

    def get_transform_calibration_fn(self):
        process_one = super().get_transform_calibration_fn()

        def transform_fn(chunk: List[Dict]):
            samples = [process_one(data, max_tokens=128, filter_bad_tokens=False) for data in chunk]
            inputs = {}
            for input_name, sample_value in samples[0].items():
                if isinstance(sample_value, torch.Tensor):
                    inputs[input_name] = torch.cat([sample[input_name] for sample in samples], dim=0)
                elif isinstance(sample_value, np.ndarray):
                    inputs[input_name] = np.concatenate([sample[input_name] for sample in samples], axis=0)
                elif isinstance(sample_value, ov.Tensor):
                    shape = sample_value.get_shape()
                    shape[0] = len(samples)
                    inputs[input_name] = ov.Tensor(sample_value.get_element_type(), shape)
                else:
                    msg = f"Failed to generate calibration set for {input_name} in type {type(sample_value)}"
                    raise RuntimeError(msg)
            if self.backend == BackendType.CUDA_TORCH:
                for input_name in inputs:
                    inputs[input_name] = torch.from_numpy(inputs[input_name]).cuda()
            return inputs

        return transform_fn

    def prepare_calibration_dataset(self):
        subset_size = self.compression_params.get("subset_size") or self.DEFAULT_SUBSET_SIZE
        dataset = (
            load_dataset("wikitext", "wikitext-2-v1", split="train", revision="b08601e")
            .filter(lambda example: len(example["text"].split()) > 256)
            .shuffle(seed=42)
            .select(range(subset_size))
            .to_list()
        )
        chunks = [dataset[i : i + self.batch_size] for i in range(0, subset_size, self.batch_size)]
        self.calibration_dataset = nncf.Dataset(chunks, self.get_transform_calibration_fn())

    def save_compressed_model(self):
        self.path_compressed_ir = self.output_model_dir / self.OV_MODEL_NAME
        if self.backend == BackendType.CUDA_TORCH:
            self.model_hf.float()
            for module in self.model_hf.nncf.modules():
                if isinstance(module, (INT8AsymmetricWeightsDecompressor, INT8SymmetricWeightsDecompressor)):
                    module.result_dtype = torch.float32
            export_from_model(
                self.model_hf, self.output_model_dir, stateful=False, compression_option="fp32", device="cuda"
            )
        else:
            super().save_compressed_model()

    def _dump_model_fp32(self):
        if self.backend == BackendType.CUDA_TORCH:
            export_from_model(
                self.model_hf, self.fp32_model_dir, stateful=False, compression_option="fp32", device="cuda"
            )
        else:
            super()._dump_model_fp32()

    def _compress(self):
        super()._compress()
        if self.backend in PT_BACKENDS:
            # This helps reproducibility but is not needed in actual use.
            for sparsifier in PTSparsifyActivationsAlgoBackend.get_sparsifiers(self.compressed_model):
                original_dtype = sparsifier.running_threshold.dtype
                sparsifier.running_threshold = sparsifier.running_threshold.half().to(original_dtype)


class ImageClassificationTimmSparsifyActivations(SAPipelineMixin, ImageClassificationTimm):
    DEFAULT_SUBSET_SIZE = 256

    def prepare_calibration_dataset(self):
        subset_size = self.compression_params.get("subset_size") or self.DEFAULT_SUBSET_SIZE
        val_dataset = torchvision.datasets.ImageFolder(
            root=self.data_dir / "imagenet" / "val", transform=self.transform
        )
        indices = np.random.default_rng(42).choice(len(val_dataset), size=subset_size, replace=False)
        subset = torch.utils.data.Subset(val_dataset, indices=indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=self.batch_size, num_workers=2, shuffle=False)
        self.calibration_dataset = nncf.Dataset(loader, self.get_transform_calibration_fn())
