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

import os
import re
import time
import traceback
from collections import OrderedDict
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
import yaml
from packaging import version

import nncf
from tests.cross_fw.shared.openvino_version import get_openvino_version
from tests.post_training.model_scope import PTQ_TEST_CASES
from tests.post_training.model_scope import WC_TEST_CASES
from tests.post_training.pipelines.base import XFAIL_SUFFIX
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.base import BaseTestPipeline
from tests.post_training.pipelines.base import ErrorReason
from tests.post_training.pipelines.base import ErrorReport
from tests.post_training.pipelines.base import RunInfo

DATA_ROOT = Path(__file__).parent / "data"


@pytest.fixture(scope="function", name="use_avx2")
def fixture_use_avx2():
    old_value = os.environ.get("ONEDNN_MAX_CPU_ISA")
    os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX2"
    if old_value is not None and old_value != "AVX2":
        print(f"Warning: ONEDNN_MAX_CPU_ISA is overriding to AVX2, was {old_value}")
    yield
    if old_value is None:
        del os.environ["ONEDNN_MAX_CPU_ISA"]
    else:
        os.environ["ONEDNN_MAX_CPU_ISA"] = old_value


@pytest.fixture(scope="session", name="data_dir")
def fixture_data(pytestconfig):
    if pytestconfig.getoption("data") is None:
        msg = "This test requires the --data argument to be specified."
        raise ValueError(msg)
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


@pytest.fixture(scope="session", name="torch_compile_validation")
def fixture_torch_compile_validation(pytestconfig):
    return pytestconfig.getoption("torch_compile_validation")


@pytest.fixture(scope="session", name="extra_columns")
def fixture_extra_columns(pytestconfig):
    return pytestconfig.getoption("extra_columns")


@pytest.fixture(scope="session", name="memory_monitor")
def fixture_memory_monitor(pytestconfig):
    return pytestconfig.getoption("memory_monitor")


def _parse_version(s: Path):
    version_str = re.search(r".*_(\d+\.\d+).(?:yaml|yml)", s.name).group(1)
    return version.parse(version_str)


def ref_data_correction(data: Dict, file_name: str):
    """
    Apply corrections from reference YAML files according current of OV version to the provided data dictionary.

    This function reads correction data from YAML files that match the given
    file name pattern (ptq|wc)_reference_data_(ov_version).yaml
    """
    ov_version = version.parse(get_openvino_version())

    for file_path in sorted(DATA_ROOT.glob(f"{file_name}_*.yaml"), key=_parse_version):
        file_ov_version = _parse_version(file_path)
        if file_ov_version > ov_version:
            break
        with file_path.open() as f:
            correction_data = yaml.safe_load(f)
        for m_name, c_data in correction_data.items():
            if m_name in data:
                data[m_name].update(c_data)
            else:
                data[m_name] = c_data
        print(f"Applied correction file {file_path}")

    return data


@pytest.fixture(scope="session", name="ptq_reference_data")
def fixture_ptq_reference_data():
    path_reference = DATA_ROOT / "ptq_reference_data.yaml"
    with path_reference.open() as f:
        data = yaml.safe_load(f)
    return ref_data_correction(data, "ptq_reference_data")


@pytest.fixture(scope="session", name="wc_reference_data")
def fixture_wc_reference_data():
    path_reference = DATA_ROOT / "wc_reference_data.yaml"
    with path_reference.open() as f:
        data = yaml.safe_load(f)
    data = ref_data_correction(data, "wc_reference_data")
    fp32_test_cases = defaultdict(dict)
    for test_case_name in data:
        if "atol" not in data[test_case_name]:
            data[test_case_name]["atol"] = 1e-4
        reported_name = test_case_name.split("_backend_")[0]
        fp32_case_name = f"{reported_name}_backend_FP32"
        fp32_test_cases[fp32_case_name]["metric_value"] = 1
        if "atol" not in fp32_test_cases[fp32_case_name]:
            fp32_test_cases[fp32_case_name]["atol"] = 1e-10
    data.update(fp32_test_cases)
    return data


@pytest.fixture(scope="session", name="ptq_result_data")
def fixture_ptq_report_data(output_dir, run_benchmark_app, pytestconfig):
    columns_to_drop = ["Num sparse activations", "Num int4"]
    yield from create_fixture_report_data(output_dir, run_benchmark_app, pytestconfig, columns_to_drop)


@pytest.fixture(scope="session", name="wc_result_data")
def fixture_wc_report_data(output_dir, run_benchmark_app, pytestconfig):
    columns_to_drop = ["Num sparse activations", "Num FQ"]
    yield from create_fixture_report_data(output_dir, run_benchmark_app, pytestconfig, columns_to_drop)


def create_fixture_report_data(output_dir, run_benchmark_app, pytestconfig, columns_to_drop):
    data: Dict[str, RunInfo] = {}

    yield data

    if data:
        if not run_benchmark_app:
            columns_to_drop.append("FPS")
        save_results(data, columns_to_drop, output_dir, pytestconfig.getoption("forked"))


def save_results(data, columns_to_drop, output_dir, is_forked):
    test_results = OrderedDict(sorted(data.items()))
    df = pd.DataFrame(v.get_result_dict() for v in test_results.values())
    df = df.drop(columns=columns_to_drop)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.csv"

    if is_forked and output_file.exists():
        # When run test with --forked to run test in separate process
        # Used in post_training_performance jobs
        df.to_csv(output_file, index=False, mode="a", header=False)
    else:
        df.to_csv(output_file, index=False)


def maybe_skip_test_case(test_model_param, run_fp32_backend, run_torch_cuda_backend, batch_size):
    if test_model_param["backend"] == BackendType.FP32 and not run_fp32_backend:
        pytest.skip("To run test for not quantized model use --fp32 argument")
    if (
        test_model_param["backend"] in [BackendType.CUDA_TORCH, BackendType.CUDA_FX_TORCH]
        and not run_torch_cuda_backend
    ):
        pytest.skip(f"To run test for {test_model_param['backend'].value} backend use --cuda argument")
    if batch_size and batch_size > 1 and test_model_param.get("batch_size", 1) == 1:
        pytest.skip("The model does not support batch_size > 1. Please use --batch-size 1.")
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
    model_name = test_model_param.get("model_name", test_case_name.split("_backend_")[0])
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


def _update_status(pipeline: BaseTestPipeline, errors: List[ErrorReason]) -> List[str]:
    """
    Updates status of the pipeline based on the errors encountered during the run.

    :param pipeline: The pipeline object containing run information.
    :param errors: List of errors encountered during the run.
    :return: List of unexpected errors.
    """
    pipeline.run_info.status = ""  # Successful status
    xfails, unexpected_errors = [], []
    for report in errors:
        xfail_reason = report.reason.value + XFAIL_SUFFIX
        if xfail_reason in pipeline.reference_data:
            xfails.append(f"XFAIL: {pipeline.reference_data[xfail_reason]} - {report.msg}")
        else:
            unexpected_errors.append(report.msg)
    if xfails:
        pipeline.run_info.status = "\n".join(xfails)
    if unexpected_errors:
        pipeline.run_info.status = "\n".join(unexpected_errors)
    return unexpected_errors


def _collect_errors(
    pipeline: BaseTestPipeline,
    exception_report: Optional[ErrorReport] = None,
) -> List[ErrorReport]:
    """
    Collects errors based on the pipeline's run information and exception happened during the run.

    :param pipeline: The pipeline object containing run information.
    :param exception_report: Optional exception report.
    :return: List of error reports.
    """
    errors = []

    if exception_report:
        errors.append(exception_report)
        return errors

    run_info = pipeline.run_info
    reference_data = pipeline.reference_data

    metric_value = run_info.metric_value
    metric_reference = reference_data.get("metric_value")
    metric_value_fp32 = reference_data.get("metric_value_fp32")

    if metric_value is not None and metric_value_fp32 is not None:
        run_info.metric_diff = round(metric_value - metric_value_fp32, 5)

    if metric_value is not None and metric_reference is not None:
        atol = reference_data.get("atol", 0.001)
        if not np.isclose(metric_value, metric_reference, atol=atol):
            status_msg = (
                f"Regression: Metric value is less than reference {metric_value} < {metric_reference}"
                if metric_value < metric_reference
                else f"Improvement: Metric value is better than reference {metric_value} > {metric_reference}"
            )
            errors.append(ErrorReport(ErrorReason.METRICS, status_msg))

    num_int4_reference = reference_data.get("num_int4")  # None means the check is skipped
    num_int8_reference = reference_data.get("num_int8")
    ref_num_sparse_activations = reference_data.get("num_sparse_activations")
    num_int4_value = run_info.num_compress_nodes.num_int4
    num_int8_value = run_info.num_compress_nodes.num_int8
    num_sparse_activations = run_info.num_compress_nodes.num_sparse_activations

    if num_int4_reference is not None and num_int4_reference != num_int4_value:
        status_msg = (
            f"Regression: The number of int4 ops is different than reference {num_int4_reference} != {num_int4_value}"
        )
        errors.append(ErrorReport(ErrorReason.NUM_COMPRESSED, status_msg))

    if num_int8_reference is not None and num_int8_reference != num_int8_value:
        status_msg = (
            f"Regression: The number of int8 ops is different than reference {num_int8_reference} != {num_int8_value}"
        )
        errors.append(ErrorReport(ErrorReason.NUM_COMPRESSED, status_msg))

    if ref_num_sparse_activations is not None and num_sparse_activations != ref_num_sparse_activations:
        status_msg = (
            f"Regression: The number of sparse activations is {num_sparse_activations}, "
            f"which differs from reference {ref_num_sparse_activations}."
        )
        errors.append(ErrorReport(ErrorReason.NUM_COMPRESSED, status_msg))

    return errors


def run_pipeline(
    test_case_name: str,
    reference_data: dict,
    test_cases: dict,
    result_data: Dict[str, RunInfo],
    output_dir: Path,
    data_dir: Optional[Path],
    no_eval: bool,
    batch_size: Optional[int],
    run_fp32_backend: bool,
    run_torch_cuda_backend: bool,
    subset_size: Optional[int],
    run_benchmark_app: bool,
    torch_compile_validation: bool,
    capsys: pytest.CaptureFixture,
    extra_columns: bool,
    memory_monitor: bool,
    use_avx2: Optional[bool] = None,
):
    pipeline, exception_report, test_model_param = None, None, None
    start_time = time.perf_counter()
    if test_case_name not in reference_data:
        msg = f"{test_case_name} does not exist in 'reference_data.yaml'"
        raise nncf.ValidationError(msg)
    test_model_param = test_cases[test_case_name]
    maybe_skip_test_case(test_model_param, run_fp32_backend, run_torch_cuda_backend, batch_size)
    pipeline_cls = test_model_param["pipeline_cls"]
    # Recalculates subset_size when subset_size is None
    if batch_size is None:
        batch_size = test_model_param.get("batch_size", 1)
    if batch_size > 1 and subset_size is None:
        subset_size = 300 // batch_size
        print(f"Update subset_size value based on provided batch_size to {subset_size}.")
    pipeline_kwargs = create_pipeline_kwargs(test_model_param, subset_size, test_case_name, reference_data)
    pipeline_kwargs.update(
        {
            "output_dir": output_dir,
            "data_dir": data_dir,
            "no_eval": no_eval,
            "run_benchmark_app": run_benchmark_app,
            "torch_compile_validation": torch_compile_validation,
            "batch_size": batch_size,
            "memory_monitor": memory_monitor,
        }
    )
    if use_avx2 is not None:
        pipeline_kwargs["use_avx2"] = use_avx2
    pipeline: BaseTestPipeline = pipeline_cls(**pipeline_kwargs)
    try:
        pipeline.run()
    except Exception as e:
        err_msg = str(e)
        if not err_msg:
            err_msg = "Unknown exception"
        exception_report = ErrorReport(ErrorReason.EXCEPTION, err_msg)
        traceback.print_exc()
    finally:
        if pipeline is not None:
            pipeline.cleanup_cache()
            run_info = pipeline.run_info

            captured = capsys.readouterr()
            write_logs(captured, pipeline)

            if extra_columns:
                pipeline.collect_data_from_stdout(captured.out)
        else:
            run_info = create_short_run_info(test_model_param, err_msg, test_case_name)
        run_info.time_total = time.perf_counter() - start_time

        errors = _collect_errors(pipeline, exception_report)
        unexpected_errors = _update_status(pipeline, errors)
        result_data[test_case_name] = run_info

        if unexpected_errors:
            pytest.fail(run_info.status)
        if run_info.status.startswith("XFAIL:"):
            pytest.xfail(run_info.status)


@pytest.mark.parametrize("test_case_name", PTQ_TEST_CASES.keys())
def test_ptq_quantization(
    ptq_reference_data: dict,
    test_case_name: str,
    data_dir: Path,
    output_dir: Path,
    ptq_result_data: Dict[str, RunInfo],
    no_eval: bool,
    batch_size: Optional[int],
    run_fp32_backend: bool,
    run_torch_cuda_backend: bool,
    subset_size: Optional[int],
    run_benchmark_app: bool,
    torch_compile_validation: bool,
    capsys: pytest.CaptureFixture,
    extra_columns: bool,
    memory_monitor: bool,
):
    run_pipeline(
        test_case_name,
        ptq_reference_data,
        PTQ_TEST_CASES,
        ptq_result_data,
        output_dir,
        data_dir,
        no_eval,
        batch_size,
        run_fp32_backend,
        run_torch_cuda_backend,
        subset_size,
        run_benchmark_app,
        torch_compile_validation,
        capsys,
        extra_columns,
        memory_monitor,
    )


@pytest.mark.parametrize("test_case_name", WC_TEST_CASES.keys())
def test_weight_compression(
    wc_reference_data: dict,
    test_case_name: str,
    output_dir: Path,
    wc_result_data: Dict[str, RunInfo],
    no_eval: bool,
    batch_size: int,
    run_fp32_backend: bool,
    run_torch_cuda_backend: bool,
    subset_size: Optional[int],
    run_benchmark_app: bool,
    capsys: pytest.CaptureFixture,
    extra_columns: bool,
    memory_monitor: bool,
    use_avx2: None,
):
    run_pipeline(
        test_case_name,
        wc_reference_data,
        WC_TEST_CASES,
        wc_result_data,
        output_dir,
        None,  # data_dir is not used in WC
        no_eval,
        batch_size,
        run_fp32_backend,
        run_torch_cuda_backend,
        subset_size,
        run_benchmark_app,
        False,  # torch_compile_validation is not used in WC
        capsys,
        extra_columns,
        memory_monitor,
        use_avx2,
    )
