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

from pathlib import Path
from typing import Optional

import pytest
import yaml

from tests.post_training.experimental.sparsify_activations.model_scope import SPARSIFY_ACTIVATIONS_TEST_CASES
from tests.post_training.pipelines.base import RunInfo
from tests.post_training.test_quantize_conformance import fixture_report_data  # noqa: F401
from tests.post_training.test_quantize_conformance import run_pipeline


@pytest.fixture(scope="session", name="sparsify_activations_reference_data")
def fixture_sparsify_activations_reference_data():
    path_reference = Path(__file__).parent / "reference_data.yaml"
    with path_reference.open() as f:
        data = yaml.safe_load(f)
    for test_case in data.values():
        test_case["atol"] = test_case.get("atol", 1e-3)
    return data


@pytest.mark.parametrize("test_case_name", SPARSIFY_ACTIVATIONS_TEST_CASES.keys())
def test_sparsify_activations(
    sparsify_activations_reference_data: dict,
    test_case_name: str,
    data_dir: Path,
    output_dir: Path,
    result_data: dict[str, RunInfo],
    no_eval: bool,
    batch_size: int,
    run_fp32_backend: bool,
    run_torch_cuda_backend: bool,
    subset_size: Optional[int],
    run_benchmark_app: bool,
    capsys: pytest.CaptureFixture,
    extra_columns: bool,
):
    run_pipeline(
        test_case_name,
        sparsify_activations_reference_data,
        SPARSIFY_ACTIVATIONS_TEST_CASES,
        result_data,
        output_dir,
        data_dir,
        no_eval,
        batch_size,
        run_fp32_backend,
        run_torch_cuda_backend,
        subset_size,
        run_benchmark_app,
        capsys,
        extra_columns,
        False,  # memory_monitor is not used in SA
    )
