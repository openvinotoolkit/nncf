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
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pytest
import yaml

import nncf
from tests.post_training.model_scope import PTQ_TEST_CASES
from tests.post_training.model_scope import WC_TEST_CASES
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.base import BaseTestPipeline
from tests.post_training.pipelines.base import RunInfo


@pytest.fixture(scope="session", name="data_dir")
def fixture_data(pytestconfig):
    if pytestconfig.getoption("data") is None:
        raise ValueError("This test requires the --data argument to be specified.")
    return Path(pytestconfig.getoption("data"))


@pytest.fixture(scope="session", name="output_dir")
def fixture_output(pytestconfig):
    return Path(pytestconfig.getoption("output"))


@pytest.fixture(scope="session", name="no_eval")
def fixture_no_eval(pytestconfig):
    return pytestconfig.getoption("no_eval")


@pytest.fixture(scope="session", name="batch_size")
def fixture_batch_size(pytestconfig):
    return pytestconfig.getoption("batch_size")


@pytest.fixture(scope="session", name="subset_size")
def fixture_subset_size(pytestconfig):
    return pytestconfig.getoption("subset_size")


@pytest.fixture(scope="session", name="run_fp32_backend")
def fixture_run_fp32_backend(pytestconfig):
    return pytestconfig.getoption("fp32")


@pytest.fixture(scope="session", name="run_torch_cuda_backend")
def fixture_run_torch_cuda_backend(pytestconfig):
    return pytestconfig.getoption("cuda")


@pytest.fixture(scope="session", name="run_benchmark_app")
def fixture_run_benchmark_app(pytestconfig):
    return pytestconfig.getoption("benchmark")


@pytest.fixture(scope="session", name="extra_columns")
def fixture_extra_columns(pytestconfig):
    return pytestconfig.getoption("extra_columns")


@pytest.fixture(scope="session", name="ptq_reference_data")
def fixture_ptq_reference_data():
    path_reference = Path(__file__).parent / "data" / "ptq_reference_data.yaml"
    with path_reference.open() as f:
        data = yaml.safe_load(f)
    return data


@pytest.fixture(scope="session", name="wc_reference_data")
def fixture_wc_reference_data():
    path_reference = Path(__file__).parent / "data" / "wc_reference_data.yaml"
    with path_reference.open() as f:
        data = yaml.safe_load(f)
        fp32_test_cases = defaultdict(dict)
        for test_case_name in data:
            if "atol" not in data[test_case_name]:
                data[test_case_name]["atol"] = 1e-5
            reported_name = test_case_name.split("_backend_")[0]
            fp32_case_name = f"{reported_name}_backend_FP32"
            fp32_test_cases[fp32_case_name]["metric_value"] = 1
            if "atol" not in fp32_test_cases[fp32_case_name]:
                fp32_test_cases[fp32_case_name]["atol"] = 1e-10
        data.update(fp32_test_cases)
    return data


@pytest.fixture(scope="session", name="ptq_result_data")
def fixture_ptq_report_data(output_dir, run_benchmark_app):
    data: Dict[str, RunInfo] = {}

    yield data

    if data:
        test_results = OrderedDict(sorted(data.items()))
        df = pd.DataFrame(v.get_result_dict() for v in test_results.values())
        if not run_benchmark_app:
            df = df.drop(columns=["FPS"])

        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "results.csv", index=False)


@pytest.fixture(scope="session", name="wc_result_data")
def fixture_wc_report_data(output_dir):
    data: Dict[str, RunInfo] = {}

    yield data

    if data:
        test_results = OrderedDict(sorted(data.items()))
        df = pd.DataFrame(v.get_result_dict() for v in test_results.values())
        df = df.drop(columns=["FPS", "Num FQ"])

        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "results.csv", index=False)


def maybe_skip_test_case(test_model_param, run_fp32_backend, run_torch_cuda_backend):
    if test_model_param["backend"] == BackendType.FP32 and not run_fp32_backend:
        pytest.skip("To run test for not quantized model use --fp32 argument")
    if test_model_param["backend"] == BackendType.CUDA_TORCH and not run_torch_cuda_backend:
        pytest.skip("To run test for CUDA_TORCH backend use --cuda argument")
    return test_model_param


def create_short_run_info(test_model_param, err_msg, test_case_name):
    if test_model_param is not None:
        run_info = RunInfo(
            model=test_model_param["reported_name"],
            backend=test_model_param["backend"],
            status=err_msg,
        )
    else:
        splitted = test_case_name.split("_backend_")
        run_info = RunInfo(
            model=splitted[0],
            backend=BackendType[splitted[1]],
            status=err_msg,
        )

    return run_info


def write_logs(captured, pipeline):
    stdout_file = pipeline.output_model_dir / "stdout.log"
    stderr_file = pipeline.output_model_dir / "stderr.log"

    stdout_file.write_text(captured.out, encoding="utf-8")
    stderr_file.write_text(captured.err, encoding="utf-8")
    return captured


def create_pipeline_kwargs(test_model_param, subset_size, test_case_name, reference_data):
    if subset_size:
        if "compression_params" not in test_model_param:
            test_model_param["compression_params"] = {}
        test_model_param["compression_params"]["subset_size"] = subset_size

    print("\n")
    print(f"Model: {test_model_param['reported_name']}")
    print(f"Backend: {test_model_param['backend']}")
    print(f"PTQ params: {test_model_param['compression_params']}")

    # Get target fp32 metric value
    model_name = test_case_name.split("_backend_")[0]
    test_reference = reference_data[test_case_name]
    test_reference["metric_value_fp32"] = reference_data[f"{model_name}_backend_FP32"]["metric_value"]

    return {
        "reported_name": test_model_param["reported_name"],
        "model_id": test_model_param["model_id"],
        "backend": test_model_param["backend"],
        "compression_params": test_model_param["compression_params"],
        "params": test_model_param.get("params"),
        "reference_data": test_reference,
    }


@pytest.mark.parametrize("test_case_name", PTQ_TEST_CASES.keys())
def test_ptq_quantization(
    ptq_reference_data: dict,
    test_case_name: str,
    data_dir: Path,
    output_dir: Path,
    ptq_result_data: Dict[str, RunInfo],
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
        if test_case_name not in ptq_reference_data:
            raise nncf.ValidationError(f"{test_case_name} does not exist in 'reference_data.yaml'")
        test_model_param = PTQ_TEST_CASES[test_case_name]
        maybe_skip_test_case(test_model_param, run_fp32_backend, run_torch_cuda_backend)
        pipeline_cls = test_model_param["pipeline_cls"]
        pipeline_kwargs = create_pipeline_kwargs(test_model_param, subset_size, test_case_name, ptq_reference_data)
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
    ptq_result_data[test_case_name] = run_info

    if err_msg:
        pytest.fail(err_msg)


@pytest.mark.parametrize("test_case_name", WC_TEST_CASES.keys())
def test_weight_compression(
    wc_reference_data: dict,
    test_case_name: str,
    data_dir: Path,
    output_dir: Path,
    wc_result_data: Dict[str, RunInfo],
    no_eval: bool,
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
        if test_case_name not in wc_reference_data:
            raise RuntimeError(f"{test_case_name} is not defined in `wc_reference_data` fixture")
        test_model_param = WC_TEST_CASES[test_case_name]
        maybe_skip_test_case(test_model_param, run_fp32_backend, run_torch_cuda_backend)
        pipeline_cls = test_model_param["pipeline_cls"]
        pipeline_kwargs = create_pipeline_kwargs(test_model_param, subset_size, test_case_name, wc_reference_data)
        pipeline_kwargs.update(
            {
                "output_dir": output_dir,
                "data_dir": data_dir,
                "no_eval": no_eval,
                "run_benchmark_app": run_benchmark_app,
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
    wc_result_data[test_case_name] = run_info

    if err_msg:
        pytest.fail(err_msg)
