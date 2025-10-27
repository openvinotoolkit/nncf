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
from functools import partial
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest
import torch
import yaml
from packaging import version

import nncf
from tests.cross_fw.shared.openvino_version import get_openvino_version
from tests.post_training.model_scope import PTQ_TEST_CASES
from tests.post_training.model_scope import WC_TEST_CASES
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.base import BaseTestPipeline
from tests.post_training.pipelines.base import ErrorReason
from tests.post_training.pipelines.base import ErrorReport
from tests.post_training.pipelines.base import RunInfo

DATA_ROOT = Path(__file__).parent / "data"


# TODO(AlexanderDokuchaev): Remove it after update optimum
torch.onnx.export = partial(torch.onnx.export, dynamo=False)

# TODO(AlexanderDokuchaev): Remove it after fix issue in optimum-intel
from optimum.exporters.tasks import TasksManager  # noqa: E402

TasksManager._TRANSFORMERS_TASKS_TO_MODEL_LOADERS["image-text-to-text"] = "AutoModelForImageTextToText"


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


def _parse_version(s: Path):
    version_str = re.search(r".*_(\d+\.\d+).(?:yaml|yml)", s.name).group(1)
    return version.parse(version_str)


def ref_data_correction(data: dict, file_name: str):
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


@pytest.fixture(scope="session", name="result_data")
def fixture_report_data(output_dir, run_benchmark_app, forked):
    data: dict[str, RunInfo] = {}
    yield data
    if data:
        columns_to_drop = ["FPS"] if not run_benchmark_app else []
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "results.csv"
        test_results = OrderedDict(sorted(data.items()))
        test_results = [v.get_result_dict() for v in test_results.values()]
        df = pd.DataFrame(test_results).drop(columns=columns_to_drop)
        if forked and output_file.exists():
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


def run_pipeline(
    test_case_name: str,
    reference_data: dict,
    test_cases: dict,
    result_data: dict[str, RunInfo],
    output_dir: Path,
    data_dir: Optional[Path],
    no_eval: bool,
    batch_size: Optional[int],
    run_fp32_backend: bool,
    run_torch_cuda_backend: bool,
    subset_size: Optional[int],
    run_benchmark_app: bool,
    capsys: pytest.CaptureFixture,
    extra_columns: bool,
    memory_monitor: bool,
):
    pipeline, exception_report, test_model_param = None, None, None
    start_time = time.perf_counter()
    if test_case_name not in reference_data:
        msg = f"{test_case_name} does not exist in 'reference_data.yaml'"
        raise nncf.ValidationError(msg)
    test_model_param = test_cases[test_case_name]
    maybe_skip_test_case(test_model_param, run_fp32_backend, run_torch_cuda_backend, batch_size)
    pipeline_cls = test_model_param["pipeline_cls"]
    pipeline_kwargs = create_pipeline_kwargs(test_model_param, subset_size, test_case_name, reference_data)
    pipeline_kwargs.update(
        {
            "output_dir": output_dir,
            "data_dir": data_dir,
            "no_eval": no_eval,
            "run_benchmark_app": run_benchmark_app,
            "batch_size": batch_size or test_model_param.get("batch_size", 1),
            "memory_monitor": memory_monitor,
        }
    )
    pipeline: BaseTestPipeline = pipeline_cls(**pipeline_kwargs)
    try:
        pipeline.run()
    except Exception as e:
        message = f"{type(e).__name__} | {str(e)}"
        exception_report = ErrorReport(ErrorReason.EXCEPTION, message)
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
            run_info = create_short_run_info(test_model_param, message, test_case_name)
        run_info.time_total = time.perf_counter() - start_time

        if exception_report:
            unexpected_errors = pipeline.update_status([exception_report])
        else:
            errors = pipeline.collect_errors()
            unexpected_errors = pipeline.update_status(errors)

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
    result_data: dict[str, RunInfo],
    no_eval: bool,
    batch_size: Optional[int],
    run_fp32_backend: bool,
    run_torch_cuda_backend: bool,
    subset_size: Optional[int],
    run_benchmark_app: bool,
    capsys: pytest.CaptureFixture,
    extra_columns: bool,
    memory_monitor: bool,
):
    run_pipeline(
        test_case_name,
        ptq_reference_data,
        PTQ_TEST_CASES,
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
        memory_monitor,
    )


@pytest.mark.parametrize("test_case_name", WC_TEST_CASES.keys())
def test_weight_compression(
    wc_reference_data: dict,
    test_case_name: str,
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
    memory_monitor: bool,
    use_avx2: None,
):
    run_pipeline(
        test_case_name,
        wc_reference_data,
        WC_TEST_CASES,
        result_data,
        output_dir,
        None,  # data_dir is not used in WC
        no_eval,
        batch_size,
        run_fp32_backend,
        run_torch_cuda_backend,
        subset_size,
        run_benchmark_app,
        capsys,
        extra_columns,
        memory_monitor,
    )
