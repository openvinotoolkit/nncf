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


import time
import traceback
from collections import OrderedDict
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pytest
import torch
import yaml
from datasets import load_dataset
from memory_profiler import memory_usage
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoModelForCausalLM

import nncf
import nncf.experimental
import nncf.experimental.torch
import nncf.experimental.torch.sparsify_activations
from nncf.experimental.torch.sparsify_activations.torch_backend import SparsifyActivationsAlgoBackend
from nncf.parameters import CompressWeightsMode
from tests.post_training.model_scope import generate_tests_scope
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.base import BaseTestPipeline
from tests.post_training.pipelines.base import RunInfo
from tests.post_training.pipelines.lm_weight_compression import LMWeightCompression
from tests.post_training.pipelines.lm_weight_compression import WCTimeStats
from tests.post_training.test_quantize_conformance import create_pipeline_kwargs
from tests.post_training.test_quantize_conformance import create_short_run_info
from tests.post_training.test_quantize_conformance import fixture_batch_size  # noqa: F401
from tests.post_training.test_quantize_conformance import fixture_data  # noqa: F401
from tests.post_training.test_quantize_conformance import fixture_extra_columns  # noqa: F401
from tests.post_training.test_quantize_conformance import fixture_no_eval  # noqa: F401
from tests.post_training.test_quantize_conformance import fixture_output  # noqa: F401
from tests.post_training.test_quantize_conformance import fixture_run_benchmark_app  # noqa: F401
from tests.post_training.test_quantize_conformance import fixture_run_fp32_backend  # noqa: F401
from tests.post_training.test_quantize_conformance import fixture_run_torch_cuda_backend  # noqa: F401
from tests.post_training.test_quantize_conformance import fixture_subset_size  # noqa: F401
from tests.post_training.test_quantize_conformance import maybe_skip_test_case
from tests.post_training.test_quantize_conformance import write_logs
from tests.torch.helpers import set_torch_seed


@pytest.fixture(scope="session", name="sparsify_activations_reference_data")
def fixture_sparsify_activations_reference_data():
    path_reference = Path(__file__).parent / "reference_data.yaml"
    with path_reference.open() as f:
        data = yaml.safe_load(f)
        fp32_test_cases = defaultdict(dict)
        for test_case_name, test_case in data.items():
            fp32_case = dict(metric_value=1.0)
            fp32_case["num_int4"] = test_case.get("num_int4", 0)
            fp32_case["num_int8"] = test_case.get("num_int8", 0)
            reported_name = test_case_name.split("_backend_")[0]
            fp32_case_name = f"{reported_name}_backend_FP32"
            fp32_test_cases[fp32_case_name] = fp32_case
        data.update(fp32_test_cases)
        for test_case in data.values():
            test_case["atol"] = test_case.get("atol", 1e-5)
    return data


@pytest.fixture(scope="session", name="sparsify_activations_result_data")
def fixture_sparsify_activations_report_data(output_dir):
    data: Dict[str, RunInfo] = {}
    yield data
    if data:
        test_results = OrderedDict(sorted(data.items()))
        df = pd.DataFrame(v.get_result_dict() for v in test_results.values())
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "results.csv", index=False)


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
                self.model_id, torch_dtype=torch.float32, device_map="cpu"
            )
            self.model = self.model_hf
        elif self.backend in [BackendType.OV, BackendType.FP32]:
            if is_stateful:
                self.fp32_model_dir = self.fp32_model_dir.parent / (self.fp32_model_dir.name + "_sf")
            if not (self.fp32_model_dir / self.OV_MODEL_NAME).exists():
                # export by model_id
                self.model_hf = OVModelForCausalLM.from_pretrained(
                    self.model_id, export=True, load_in_8bit=False, compile=False, stateful=is_stateful
                )
            else:
                # no export, load from IR. Applicable for sequential run of test cases in local environment.
                self.model_hf = OVModelForCausalLM.from_pretrained(
                    self.fp32_model_dir, trust_remote_code=True, load_in_8bit=False, compile=False, stateful=is_stateful
                )
            self.model = self.model_hf.model
        else:
            raise RuntimeError(f"backend={self.backend.value} is not supported.")

        if not (self.fp32_model_dir / self.OV_MODEL_NAME).exists():
            self._dump_model_fp32()

    def prepare_calibration_dataset(self):
        dataset = load_dataset("wikitext", "wikitext-2-v1", split="train", revision="b08601e")
        dataset = dataset.filter(lambda example: len(example["text"].split()) > 256)
        dataset = dataset.select(range(64))
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


SPARSIFY_ACTIVATIONS_MODELS = [
    {
        "reported_name": "tinyllama",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMSparsifyActivations,
        "compression_params": None,
        "backends": [BackendType.FP32],
    },
    {
        "reported_name": "tinyllama_ffn_sparse20",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMSparsifyActivations,
        "compression_params": {
            "compress_weights": None,
            "sparsify_activations": {
                "target_sparsity_by_scope": {
                    "{re}up_proj": 0.2,
                    "{re}gate_proj": 0.2,
                    "{re}down_proj": 0.2,
                }
            },
        },
        "backends": [BackendType.TORCH],
    },
    {
        "reported_name": "tinyllama_int8_asym_data_free_ffn_sparse20",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMSparsifyActivations,
        "compression_params": {
            "compress_weights": {
                "mode": CompressWeightsMode.INT8_ASYM,
            },
            "sparsify_activations": {
                "target_sparsity_by_scope": {
                    "{re}up_proj": 0.2,
                    "{re}gate_proj": 0.2,
                    "{re}down_proj": 0.2,
                }
            },
        },
        "backends": [BackendType.TORCH],
    },
]


SPARSIFY_ACTIVATIONS_TEST_CASES = generate_tests_scope(SPARSIFY_ACTIVATIONS_MODELS)


@pytest.mark.parametrize("test_case_name", SPARSIFY_ACTIVATIONS_TEST_CASES.keys())
def test_sparsify_activations(
    sparsify_activations_reference_data: dict,
    test_case_name: str,
    data_dir: Path,
    output_dir: Path,
    sparsify_activations_result_data: Dict[str, RunInfo],
    no_eval: bool,
    batch_size: int,
    run_fp32_backend: bool,
    run_torch_cuda_backend: bool,
    subset_size: Optional[int],
    run_benchmark_app: bool,
    capsys: pytest.CaptureFixture,
    extra_columns: bool,
):
    pipeline = None
    err_msg = None
    test_model_param = None
    start_time = time.perf_counter()
    try:
        if test_case_name not in sparsify_activations_reference_data:
            raise RuntimeError(f"{test_case_name} is not defined in `sparsify_activations_reference_data` fixture")
        test_model_param = SPARSIFY_ACTIVATIONS_TEST_CASES[test_case_name]
        maybe_skip_test_case(test_model_param, run_fp32_backend, run_torch_cuda_backend, batch_size)
        pipeline_cls = test_model_param["pipeline_cls"]
        pipeline_kwargs = create_pipeline_kwargs(
            test_model_param, subset_size, test_case_name, sparsify_activations_reference_data
        )
        pipeline_kwargs.update(
            {
                "output_dir": output_dir,
                "data_dir": data_dir,
                "no_eval": no_eval,
                "run_benchmark_app": run_benchmark_app,
                "batch_size": batch_size,
            }
        )
        pipeline: BaseTestPipeline = pipeline_cls(**pipeline_kwargs)
        pipeline.run()
    except Exception as e:
        err_msg = str(e)
        traceback.print_exc()

    if pipeline is not None:
        pipeline.cleanup_cache()
        run_info = pipeline.run_info
        if err_msg:
            run_info.status = f"{run_info.status} | {err_msg}" if run_info.status else err_msg

        captured = capsys.readouterr()
        write_logs(captured, pipeline)

        if extra_columns:
            pipeline.collect_data_from_stdout(captured.out)
    else:
        run_info = create_short_run_info(test_model_param, err_msg, test_case_name)

    run_info.time_total = time.perf_counter() - start_time
    sparsify_activations_result_data[test_case_name] = run_info

    if err_msg:
        pytest.fail(err_msg)
