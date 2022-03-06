"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import sys
import datetime

import pytest
from pathlib import Path

from tests.tensorflow.test_sota_checkpoints import BG_COLOR_GREEN_HEX
from tests.tensorflow.test_sota_checkpoints import BG_COLOR_RED_HEX
from tests.tensorflow.test_sota_checkpoints import DATASET_TYPE_AVAILABILITY
from tests.tensorflow.test_sota_checkpoints import EVAL_SCRIPT_NAME_MAP
from tests.tensorflow.test_sota_checkpoints import EvalRunParamsStruct
from tests.tensorflow.test_sota_checkpoints import RunTest
from tests.common.helpers import PROJECT_ROOT

DIFF_TARGET_INIT_MIN_GLOBAL = -0.3
DIFF_TARGET_INIT_MAX_GLOBAL = 0.3

MODE = 'TF2'


class TestInitialization(RunTest):
    @staticmethod
    def update_tag_text(tag, text):
        with tag('p'):
            with tag('span', style=f'Background-color: #{BG_COLOR_GREEN_HEX}'):
                text('Thresholds for Measured after initialization and Expected are passed')
        with tag('p'):
            with tag('span', style=f'Background-color: #{BG_COLOR_RED_HEX}'):
                text('Thresholds for Measured after initialization and Expected are failed')

    @staticmethod
    def threshold_check(is_ok, diff_target):
        color = BG_COLOR_GREEN_HEX
        within_thresholds = True
        if is_ok:
            if not DIFF_TARGET_INIT_MIN_GLOBAL < diff_target < DIFF_TARGET_INIT_MAX_GLOBAL:
                color = BG_COLOR_RED_HEX
                within_thresholds = False
        else:
            color = BG_COLOR_RED_HEX
            within_thresholds = False
        return color, within_thresholds

    @staticmethod
    def make_table_row(expected,  metric, diff, error_message, metrics_type_, key):
        if metric is not None:
            row = [str(key), str(metrics_type_), str(expected), str(metric), str(diff), str('-')]
        else:
            row = [str(key), str(metrics_type_), str(expected), str('Not executed'), str('-'),
                   str(error_message)]
        return row

    @pytest.mark.init
    @pytest.mark.parametrize('eval_test_struct', RunTest.param_list,
                             ids=RunTest.ids_list)
    def test_init(self, sota_checkpoints_dir, sota_data_dir, eval_test_struct: EvalRunParamsStruct):
        # pylint: disable=too-many-branches
        if sota_data_dir is None:
            pytest.skip('Path to datasets is not set')

        if eval_test_struct.resume_file_ is None or eval_test_struct.sample_type_ == 'segmentation':
            pytest.skip('Skip initialization run for the full-precision and all segmentation models')

        sample_type = eval_test_struct.sample_type_
        log_dir = pytest.metrics_dump_path / 'logs'
        metric_file_name = self.get_metric_file_name(model_name=eval_test_struct.model_name_)
        metrics_dump_file_path = pytest.metrics_dump_path / metric_file_name
        if MODE == 'TF2':
            cmd = (f'{sys.executable} examples/tensorflow/{sample_type}/{EVAL_SCRIPT_NAME_MAP[sample_type]} '
                   f'-m test --config {eval_test_struct.config_name_} '
                   f'--data {sota_data_dir}/{eval_test_struct.dataset_type_}/{eval_test_struct.dataset_name_}/ '
                   f'--log-dir={log_dir} --metrics-dump {metrics_dump_file_path}')
            if eval_test_struct.weights_:
                cmd += f' --weights {os.path.join(sota_checkpoints_dir, eval_test_struct.weights_)}'
            if DATASET_TYPE_AVAILABILITY[sample_type]:
                cmd += f' --dataset-type {eval_test_struct.dataset_type_}'
            cmd += ' --seed 1'
        else:
            cmd = (f'{sys.executable} examples/tensorflow/{sample_type}/main.py -m test '
                   f'--config {eval_test_struct.config_name_} '
                   f'--data {sota_data_dir}/{eval_test_struct.dataset_name_}/ --log-dir={log_dir}')
        if eval_test_struct.batch_:
            cmd += f' -b {eval_test_struct.batch_}'

        cmd = cmd + f' --metrics-dump {metrics_dump_file_path}'
        exit_code, err_str = self.run_cmd(cmd, cwd=PROJECT_ROOT)
        is_ok = (exit_code == 0 and metrics_dump_file_path.exists())

        if is_ok:
            metric_value = self.read_metric(str(metrics_dump_file_path))
            diff_target = round((metric_value - eval_test_struct.expected_init_), 2)
        else:
            metric_value = None
            diff_target = None

        self.row_dict[eval_test_struct.model_name_] = self.make_table_row(eval_test_struct.expected_init_,
                                                                          metric_value,
                                                                          diff_target,
                                                                          err_str,
                                                                          eval_test_struct.metric_type_,
                                                                          eval_test_struct.model_name_)
        retval = self.threshold_check(is_ok, diff_target)

        self.color_dict[eval_test_struct.model_name_], is_accuracy_within_thresholds = retval
        assert is_accuracy_within_thresholds


# pylint:disable=line-too-long
@pytest.fixture(autouse=True, scope='class')
def make_metrics_dump_path(metrics_dump_dir):
    if pytest.metrics_dump_path is None:
        data = datetime.datetime.now()
        data_stamp = '_'.join([str(getattr(data, atr)) for atr in ['year', 'month', 'day', 'hour', 'minute', 'second']])
        pytest.metrics_dump_path = PROJECT_ROOT / 'test_results' / 'metrics_dump_' / f'{data_stamp}'
    else:
        pytest.metrics_dump_path = Path(pytest.metrics_dump_path)
    assert not os.path.isdir(pytest.metrics_dump_path) or not os.listdir(pytest.metrics_dump_path), \
        f'metrics_dump_path dir should be empty: {pytest.metrics_dump_path}'
    print(f'metrics_dump_path: {pytest.metrics_dump_path}')


@pytest.fixture(autouse=True, scope='class')
def results(sota_data_dir):
    yield
    if sota_data_dir:
        RunTest.write_common_metrics_file(per_model_metric_file_dump_path=pytest.metrics_dump_path)
        header = ['Model', 'Metrics type', 'Expected Init', 'Measured Init', 'Diff Expected-Measured Init', 'Error Init']
        TestInitialization().write_results_table(header, pytest.metrics_dump_path)
