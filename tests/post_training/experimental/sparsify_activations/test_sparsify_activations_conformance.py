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


import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pytest
import yaml

from tests.post_training.experimental.sparsify_activations.model_scope import SPARSIFY_ACTIVATIONS_TEST_CASES
from tests.post_training.experimental.sparsify_activations.pipelines import SARunInfo
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.base import BaseTestPipeline
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


@pytest.fixture(scope="session", name="sparsify_activations_reference_data")
def fixture_sparsify_activations_reference_data():
    path_reference = Path(__file__).parent / "reference_data.yaml"
    with path_reference.open() as f:
        data = yaml.safe_load(f)
    for test_case in data.values():
        test_case["atol"] = test_case.get("atol", 1e-3)
    return data


@pytest.fixture(scope="session", name="sparsify_activations_result_data")
def fixture_sparsify_activations_report_data(output_dir):
    data: Dict[str, SARunInfo] = {}
    yield data
    if data:
        test_results = OrderedDict(sorted(data.items()))
        df = pd.DataFrame(v.get_result_dict() for v in test_results.values())
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "results.csv", index=False)


def create_pipeline_kwargs(
    test_model_param: Dict,
    subset_size,
    test_case_name: str,
    reference_data: Dict[str, Dict],
    fp32_model_params: Dict[str, Dict],
):
    if subset_size:
        if "compression_params" not in test_model_param:
            test_model_param["compression_params"] = {}
        test_model_param["compression_params"]["subset_size"] = subset_size

    print("\n")
    print(f"Model: {test_model_param['reported_name']}")
    print(f"Backend: {test_model_param['backend']}")
    print(f"Comprssion params: {test_model_param['compression_params']}")

    # Get target fp32 metric value
    model_id = test_model_param["model_id"]
    fp32_test_case_name = fp32_model_params[model_id]["reported_name"] + f"_backend_{BackendType.FP32.value}"
    test_reference = reference_data[test_case_name]
    test_reference["metric_value_fp32"] = reference_data[fp32_test_case_name]["metric_value"]

    return {
        "reported_name": test_model_param["reported_name"],
        "model_id": test_model_param["model_id"],
        "backend": test_model_param["backend"],
        "compression_params": test_model_param["compression_params"],
        "params": test_model_param.get("params"),
        "reference_data": test_reference,
    }


@pytest.mark.parametrize("test_case_name", SPARSIFY_ACTIVATIONS_TEST_CASES.keys())
def test_sparsify_activations(
    sparsify_activations_reference_data: dict,
    test_case_name: str,
    data_dir: Path,
    output_dir: Path,
    sparsify_activations_result_data: Dict[str, SARunInfo],
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
        fp32_model_params = {
            tc["model_id"]: tc for tc in SPARSIFY_ACTIVATIONS_TEST_CASES.values() if tc["backend"] == BackendType.FP32
        }
        pipeline_cls = test_model_param["pipeline_cls"]
        pipeline_kwargs = create_pipeline_kwargs(
            test_model_param, subset_size, test_case_name, sparsify_activations_reference_data, fp32_model_params
        )
        calibration_batch_size = batch_size or test_model_param.get("batch_size", 1)
        pipeline_kwargs.update(
            {
                "output_dir": output_dir,
                "data_dir": data_dir,
                "no_eval": no_eval,
                "run_benchmark_app": run_benchmark_app,
                "batch_size": calibration_batch_size,
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
