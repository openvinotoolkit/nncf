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


import time
import traceback
from pathlib import Path

import pytest
import yaml

from tests.post_training.model_scope import TEST_CASES
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.base import RunInfo


@pytest.fixture(scope="session", name="data")
def fixture_data(pytestconfig):
    return pytestconfig.getoption("data")


@pytest.fixture(scope="session", name="output")
def fixture_output(pytestconfig):
    return pytestconfig.getoption("output")


@pytest.fixture(scope="session", name="result")
def fixture_result(pytestconfig):
    return pytestconfig.test_results


@pytest.fixture(scope="session", name="no_eval")
def fixture_no_eval(pytestconfig):
    return pytestconfig.getoption("no_eval")


def read_reference_data():
    path_reference = Path(__file__).parent / "reference_data.yaml"
    with path_reference.open() as f:
        data = yaml.safe_load(f)
    return data


REFERENCE_DATA = read_reference_data()


@pytest.mark.parametrize("test_case_name", TEST_CASES.keys())
def test_ptq_quantization(test_case_name, data, output, result, no_eval):
    pipeline = None
    err_msg = None
    test_model_param = None
    start_time = time.perf_counter()

    try:
        if test_case_name not in REFERENCE_DATA:
            raise RuntimeError(f"{test_case_name} does not exist in 'reference_data.yaml'")

        test_model_param = TEST_CASES[test_case_name]
        pipeline_cls = test_model_param["pipeline_cls"]

        print("\n")
        print(f"Model: {test_model_param['reported_name']}")
        print(f"Backend: {test_model_param['backend']}")
        print(f"PTQ params: {test_model_param['ptq_params']}")

        pipeline_kwargs = {
            "reported_name": test_model_param["reported_name"],
            "model_id": test_model_param["model_id"],
            "backend": test_model_param["backend"],
            "ptq_params": test_model_param["ptq_params"],
            "params": test_model_param.get("params"),
            "output_dir": output,
            "data_dir": data,
            "reference_data": REFERENCE_DATA[test_case_name],
            "no_eval": no_eval,
        }

        pipeline = pipeline_cls(**pipeline_kwargs)
        pipeline.run()

    except Exception as e:
        err_msg = str(e)
        traceback.print_exc()

    if pipeline is not None:
        run_info = pipeline.get_run_info()
        if err_msg:
            run_info.status = f"{run_info.status} | {err_msg}" if run_info.status else err_msg
    else:
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

    run_info.time_total = time.perf_counter() - start_time
    result[test_case_name] = run_info.get_result_dict()

    if err_msg:
        pytest.fail(err_msg)
