"""
 Copyright (c) 2021 Intel Corporation
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
import json
import sys
import csv
import datetime
from typing import Tuple, List, Optional

import tensorflow as tf
import pytest
import subprocess
import re
import shlex
from prettytable import PrettyTable
from collections import OrderedDict
from yattag import Doc
from pathlib import Path
from beta.tests.conftest import TEST_ROOT, PROJECT_ROOT

BG_COLOR_GREEN_HEX = 'ccffcc'
BG_COLOR_YELLOW_HEX = 'ffffcc'
BG_COLOR_RED_HEX = 'ffcccc'

DIFF_TARGET_MIN_GLOBAL = -0.1
DIFF_TARGET_MAX_GLOBAL = 0.1
DIFF_FP32_MIN_GLOBAL = -1.0
DIFF_FP32_MAX_GLOBAL = 0.1

MODE = "TF2"

EVAL_SCRIPT_NAME_MAP = {
    "classification": "main.py",
    "object_detection": "main.py",
    "segmentation": "evaluation.py",
}

PRETRAINED_PARAM_AVAILABILITY = {
    "classification": True,
    "object_detection": False,
    "segmentation": False,
}

DATASET_TYPE_AVAILABILITY = {
    "classification": True,
    "object_detection": True,
    "segmentation": False,
}

num_gpus = len(tf.config.list_physical_devices('GPU'))
BATCH_COEFF = num_gpus if num_gpus else 1

class EvalRunParamsStruct:
    def __init__(self,
                 config_name_: str,
                 reference_: Optional[str],
                 expected_: float,
                 metric_type_: str,
                 dataset_name_: str,
                 dataset_type_: str,
                 sample_type_: str,
                 resume_file_: str,
                 weights_: str,
                 batch_: int,
                 mean_val_: Optional[str],
                 scale_val_: Optional[str],
                 diff_fp32_min_: float,
                 diff_fp32_max_: float,
                 model_name_: str,
                 diff_target_min_: float,
                 diff_target_max_: float
                 ):
        self.config_name_ = config_name_
        self.reference_ = reference_
        self.expected_ = expected_
        self.metric_type_ = metric_type_
        self.dataset_name_ = dataset_name_
        self.dataset_type_ = dataset_type_
        self.sample_type_ = sample_type_
        self.resume_file_ = resume_file_
        self.weights_ = weights_
        self.batch_ = batch_
        self.mean_val_ = mean_val_
        self.scale_val_ = scale_val_
        self.diff_fp32_min_ = diff_fp32_min_
        self.diff_fp32_max_ = diff_fp32_max_
        self.model_name_ = model_name_
        self.diff_target_min_ = diff_target_min_
        self.diff_target_max_ = diff_target_max_


class TestSotaCheckpoints:
    param_list = []
    train_param_list = []
    ids_list = []
    train_ids_list = []
    row_dict = OrderedDict()
    color_dict = OrderedDict()
    ref_fp32_dict = OrderedDict()
    test = None

    @staticmethod
    def get_metric_file_name(model_name: str):
        return "{}.metrics.json".format(model_name)


    if MODE == "TF2":
        CMD_FORMAT_STRING = "{} examples/tensorflow/{sample_type}/{eval_script_name} -m {} --config {conf} \
         --data {dataset}/{data_type}/{data_name}/ --log-dir={log_dir} --metrics-dump \
          {metrics_dump_file_path}"
    else:
        CMD_FORMAT_STRING = "{} examples/tensorflow/{sample_type}/main.py -m {} --config {conf} \
         --data {dataset}/{data_name}/ --log-dir={log_dir} --metrics-dump \
          {metrics_dump_file_path}"

    @staticmethod
    def run_cmd(comm: str, cwd: str) -> Tuple[int, str]:
        print()
        print(comm)
        print()
        com_line = shlex.split(comm)

        env = os.environ.copy()
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] += ":" + str(PROJECT_ROOT)
        else:
            env["PYTHONPATH"] = str(PROJECT_ROOT)
        result = subprocess.Popen(com_line, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  cwd=cwd, env=env)
        exit_code = result.poll()

        def process_line(decoded_line: str, error_lines: List):
            if re.search('Error|(No module named)', decoded_line):
                # WA for tensorboardX multiprocessing bug (https://github.com/lanpa/tensorboardX/issues/598)
                if not re.search('EOFError', decoded_line):
                    error_lines.append(decoded_line)
            if decoded_line != "":
                print(decoded_line)

        error_lines = []
        while exit_code is None:
            decoded_line = result.stdout.readline().decode('utf-8').strip()
            process_line(decoded_line, error_lines)
            exit_code = result.poll()

        # The process may exit before the first process_line is executed, handling this case here
        outs, _ = result.communicate()
        remaining_lines = outs.decode('utf-8').strip().split('\n')
        for output_line in remaining_lines:
            process_line(output_line, error_lines)

        err_string = "\n".join(error_lines) if error_lines else None
        return exit_code, err_string

    @staticmethod
    def get_onnx_model_file_path(name):
        onnx_name = PROJECT_ROOT / "onnx" / str(name + ".onnx")
        return onnx_name

    @staticmethod
    def make_table_row(test, expected_, metrics_type_, key, error_message, metric, diff_target,
                       fp32_metric_=None, diff_fp32=None, metric_type_from_json=None):
        TestSotaCheckpoints.test = test
        if fp32_metric_ is None:
            fp32_metric_ = "-"
            diff_fp32 = "-"
        if metric_type_from_json and fp32_metric_ != "-":
            fp32_metric_ = str("({})".format(fp32_metric_))
        if metric is not None:
            if test == 'eval':
                row = [str(key), str(metrics_type_), str(expected_), str(metric), str(fp32_metric_), str(diff_fp32),
                       str(diff_target), str("-")]
            else:
                row = [str(key), str(metrics_type_), str(expected_), str(metric), str(diff_target), str("-")]
        else:
            if test == 'eval':
                row = [str(key), str(metrics_type_), str(expected_), str("Not executed"), str(fp32_metric_),
                       str("-"), str("-"), str(error_message)]
            else:
                row = [str(key), str(metrics_type_), str(expected_), str("Not executed"), str("-"), str(error_message)]
        return row

    @staticmethod
    def write_error_in_csv(error_message, filename):
        with open(f'{filename}.csv', 'w', newline='') as csvfile:
            fieldnames = ['model', 'launcher', 'device', 'dataset', 'tags', 'metric_name', 'metric_type',
                          'metric_value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'model': filename, 'launcher': '-', 'device': '-', 'dataset': '-', 'tags': '-',
                             'metric_name': '-', 'metric_type': '-', 'metric_value': error_message})

    def write_results_table(self, init_table_string, path):
        result_table = PrettyTable()
        result_table.field_names = init_table_string
        for key in self.row_dict:
            result_table.add_row(self.row_dict[key])
        print()
        print(result_table)

        doc, tag, text = Doc().tagtext()
        doc.asis('<!DOCTYPE html>')
        with tag('p'):
            text('legend: ')
        with tag('p'):
            with tag('span', style="Background-color: #{}".format(BG_COLOR_GREEN_HEX)):
                text('Thresholds for FP32 and Expected are passed')
        with tag('p'):
            with tag('span', style="Background-color: #{}".format(BG_COLOR_YELLOW_HEX)):
                text('Thresholds for Expected is failed, but for FP32 passed')
        with tag('p'):
            with tag('span', style="Background-color: #{}".format(BG_COLOR_RED_HEX)):
                text('Thresholds for FP32 and Expected are failed')
        with tag('p'):
            text('If Reference FP32 value in parentheses, it takes from "target" field of .json file')
        with tag('table', border="1", cellpadding="5", style="border-collapse: collapse; border: 1px solid;"):
            with tag('tr'):
                for i in init_table_string:
                    with tag('td'):
                        text(i)
            for key in self.row_dict:
                with tag('tr', bgcolor='{}'.format(self.color_dict[key])):
                    for i in self.row_dict[key]:
                        if i is None:
                            i = '-'
                        with tag('td'):
                            text(i)
        print("Write results at ", path / "results.html")
        f = open(path / 'results.html', 'w')
        f.write(doc.getvalue())
        f.close()

    @staticmethod
    def threshold_check(is_ok, diff_target, diff_fp32_min_=None, diff_fp32_max_=None, fp32_metric=None,
                        diff_fp32=None, diff_target_min=None, diff_target_max=None):
        color = BG_COLOR_RED_HEX
        within_thresholds = False
        if not diff_target_min:
            diff_target_min = DIFF_TARGET_MIN_GLOBAL
        if not diff_target_max:
            diff_target_max = DIFF_TARGET_MAX_GLOBAL
        if not diff_fp32_min_:
            diff_fp32_min_ = DIFF_FP32_MIN_GLOBAL
        if not diff_fp32_max_:
            diff_fp32_max_ = DIFF_FP32_MAX_GLOBAL
        if is_ok:
            if fp32_metric is not None:
                if diff_fp32_min_ < diff_fp32 < diff_fp32_max_ and diff_target_min < diff_target < diff_target_max:
                    color = BG_COLOR_GREEN_HEX
                    within_thresholds = True
                elif diff_fp32_min_ < diff_fp32 < diff_fp32_max_:
                    color = BG_COLOR_YELLOW_HEX
            elif diff_target_min < diff_target < diff_target_max:
                color = BG_COLOR_GREEN_HEX
                within_thresholds = True
        return color, within_thresholds

    # pylint:disable=unused-variable
    @staticmethod
    def write_common_metrics_file(per_model_metric_file_dump_path: Path):
        metric_value = OrderedDict()
        for root, dirs, files in os.walk(per_model_metric_file_dump_path):
            for file in files:
                metric_file_path = per_model_metric_file_dump_path / file
                with open(str(metric_file_path)) as metric_file:
                    metrics = json.load(metric_file)
                model_name = str(file).split('.')[0]
                metric_value[model_name] = metrics['Accuracy']
                common_metrics_file_path = per_model_metric_file_dump_path / 'metrics.json'
                if common_metrics_file_path.is_file():
                    data = json.loads(common_metrics_file_path.read_text(encoding='utf-8'))
                    data.update(metric_value)
                    common_metrics_file_path.write_text(json.dumps(data, indent=4), encoding='utf-8')
                else:
                    with open(str(common_metrics_file_path), 'w') as outfile:
                        json.dump(metric_value, outfile)
                dirs.clear()

    @staticmethod
    def read_metric(metric_file_name: str):
        with open(metric_file_name) as metric_file:
            metrics = json.load(metric_file)
        return metrics['Accuracy']

    sota_eval_config = json.load(open('{}/sota_checkpoints_eval.json'.format(TEST_ROOT)),
                                 object_pairs_hook=OrderedDict)
    for sample_type_ in sota_eval_config:
        datasets = sota_eval_config[sample_type_]
        for dataset_name in datasets:
            model_dict = datasets[dataset_name].get('topologies')
            for model_name in model_dict:
                config_name = model_dict[model_name].get('config', {})
                weights = None
                if model_dict[model_name].get("weights", {}):
                    weights = model_dict[model_name].get("weights", {})
                reference = None
                if model_dict[model_name].get('reference', {}):
                    reference = model_dict[model_name].get('reference', {})
                else:
                    ref_fp32_dict[model_name] = model_dict[model_name].get('target', {})
                expected = model_dict[model_name].get('target', {})
                metric_type = model_dict[model_name].get('metric_type', {})
                if model_dict[model_name].get('resume', {}):
                    resume_file = model_dict[model_name].get('resume', {})
                else:
                    resume_file = None
                if model_dict[model_name].get('batch_per_gpu', {}):
                    global_batch = model_dict[model_name]['batch_per_gpu'] * BATCH_COEFF
                else:
                    global_batch = None
                if model_dict[model_name].get('mean_value', {}):
                    mean_val = model_dict[model_name].get('mean_value', {})
                else:
                    mean_val = '[123.675,116.28,103.53]'
                if model_dict[model_name].get('scale_value', {}):
                    scale_val = model_dict[model_name].get('scale_value', {})
                else:
                    scale_val = '[58.4795,57.1429,57.4713]'
                diff_fp32_min = model_dict[model_name].get('diff_fp32_min') if not None else None
                diff_fp32_max = model_dict[model_name].get('diff_fp32_max') if not None else None
                diff_target_min = model_dict[model_name].get('diff_target_min') if not None else None
                diff_target_max = model_dict[model_name].get('diff_target_max') if not None else None
                for dataset_type in datasets[dataset_name].get('dataset_types'):
                    # TODO(Evgeny Tsykunov): add TFRecords support for YOLOv4
                    if 'yolo_v4' in model_name and dataset_type == 'tfrecords':
                        continue

                    # Change model name to keep dataset version
                    model_name_with_datatype = model_name + '_' + dataset_type
                    param_list.append(EvalRunParamsStruct(config_name_=config_name,
                                                          reference_=reference,
                                                          expected_=expected,
                                                          metric_type_=metric_type,
                                                          dataset_name_=dataset_name,
                                                          dataset_type_=dataset_type,
                                                          sample_type_=sample_type_,
                                                          resume_file_=resume_file,
                                                          weights_=weights,
                                                          batch_=global_batch,
                                                          mean_val_=mean_val,
                                                          scale_val_=scale_val,
                                                          diff_fp32_min_=diff_fp32_min,
                                                          diff_fp32_max_=diff_fp32_max,
                                                          model_name_=model_name_with_datatype,
                                                          diff_target_min_=diff_target_min,
                                                          diff_target_max_=diff_target_max))
                    ids_list.append(model_name_with_datatype)
                    if model_dict[model_name].get('compression_description', {}):
                        train_param_list.append((config_name,
                                                 expected,
                                                 metric_type,
                                                 dataset_name,
                                                 sample_type_,
                                                 model_name))
                        train_ids_list.append(model_name)

    @pytest.mark.parametrize("eval_test_struct", param_list,
                             ids=ids_list)
    def test_eval(self, sota_checkpoints_dir, sota_data_dir, eval_test_struct: EvalRunParamsStruct):
        # pylint: disable=too-many-branches
        if sota_data_dir is None:
            pytest.skip('Path to datasets is not set')
        test = "eval"
        sample_type = eval_test_struct.sample_type_
        metric_file_name = self.get_metric_file_name(model_name=eval_test_struct.model_name_)
        metrics_dump_file_path = pytest.metrics_dump_path / metric_file_name
        log_dir = pytest.metrics_dump_path / "logs"
        if MODE == 'TF2':
            cmd = self.CMD_FORMAT_STRING.format(sys.executable, 'test', conf=eval_test_struct.config_name_,
                                                dataset=sota_data_dir,
                                                data_name=eval_test_struct.dataset_name_,
                                                data_type=eval_test_struct.dataset_type_,
                                                sample_type=sample_type,
                                                eval_script_name=EVAL_SCRIPT_NAME_MAP[sample_type],
                                                metrics_dump_file_path=metrics_dump_file_path, log_dir=log_dir)
            if eval_test_struct.weights_:
                cmd += " --weights {}".format(os.path.join(sota_checkpoints_dir, eval_test_struct.weights_))
            if DATASET_TYPE_AVAILABILITY[sample_type]:
                cmd += " --dataset-type {}".format(eval_test_struct.dataset_type_)
        else:
            cmd = self.CMD_FORMAT_STRING.format(sys.executable, 'test', conf=eval_test_struct.config_name_,
                                                dataset=sota_data_dir,
                                                data_name=eval_test_struct.dataset_name_,
                                                sample_type=eval_test_struct.sample_type_,
                                                metrics_dump_file_path=metrics_dump_file_path, log_dir=log_dir)
        if eval_test_struct.resume_file_:
            resume_file_path = sota_checkpoints_dir + '/' + eval_test_struct.resume_file_
            cmd += " --resume {}".format(resume_file_path)
        else:
            if MODE != "TF2" or PRETRAINED_PARAM_AVAILABILITY[sample_type]:
                cmd += " --pretrained"
        if eval_test_struct.batch_:
            cmd += " -b {}".format(eval_test_struct.batch_)
        exit_code, err_str = self.run_cmd(cmd, cwd=PROJECT_ROOT)

        is_ok = (exit_code == 0 and metrics_dump_file_path.exists())
        if is_ok:
            metric_value = self.read_metric(str(metrics_dump_file_path))
        else:
            metric_value = None

        fp32_metric = None
        metric_type_from_json = False
        if eval_test_struct.reference_ is not None:
            fp32_metric = self.ref_fp32_dict[str(eval_test_struct.reference_)]
            metric_type_from_json = True
            dataset_type_postfix = ''
            if MODE == 'TF2':
                dataset_type_postfix = '_' + eval_test_struct.dataset_type_
            reference_metric_file_path = \
                pytest.metrics_dump_path / self.get_metric_file_name(eval_test_struct.reference_ +
                                                                     dataset_type_postfix)
            if os.path.exists(reference_metric_file_path):
                with open(str(reference_metric_file_path)) as ref_metric:
                    metrics = json.load(ref_metric)
                if metrics['Accuracy'] != 0:
                    fp32_metric = metrics['Accuracy']
                    metric_type_from_json = False
            else:
                metric_type_from_json = True

        if is_ok:
            diff_target = round((metric_value - eval_test_struct.expected_), 2)
            diff_fp32 = round((metric_value - fp32_metric), 2) if fp32_metric is not None else None
        else:
            diff_target = None
            diff_fp32 = None

        self.row_dict[eval_test_struct.model_name_] = self.make_table_row(test, eval_test_struct.expected_,
                                                                          eval_test_struct.metric_type_,
                                                                          eval_test_struct.model_name_,
                                                                          err_str,
                                                                          metric_value,
                                                                          diff_target,
                                                                          fp32_metric,
                                                                          diff_fp32,
                                                                          metric_type_from_json)
        retval = self.threshold_check(is_ok,
                                      diff_target,
                                      eval_test_struct.diff_fp32_min_,
                                      eval_test_struct.diff_fp32_max_,
                                      fp32_metric,
                                      diff_fp32,
                                      eval_test_struct.diff_target_min_,
                                      eval_test_struct.diff_target_max_)

        self.color_dict[eval_test_struct.model_name_], is_accuracy_within_thresholds = retval
        assert is_accuracy_within_thresholds


Tsc = TestSotaCheckpoints


@pytest.fixture(autouse=True, scope="class")
def make_metrics_dump_path(metrics_dump_dir):
    if pytest.metrics_dump_path is None:
        data = datetime.datetime.now()
        pytest.metrics_dump_path = PROJECT_ROOT / "test_results" / "metrics_dump_" \
            f"{'_'.join([str(getattr(data, atr)) for atr in ['year', 'month', 'day', 'hour', 'minute', 'second']])}"
    else:
        pytest.metrics_dump_path = Path(pytest.metrics_dump_path)
    assert not os.path.isdir(pytest.metrics_dump_path) or not os.listdir(pytest.metrics_dump_path), \
        f"metrics_dump_path dir should be empty: {pytest.metrics_dump_path}"
    print(f"metrics_dump_path: {pytest.metrics_dump_path}")


@pytest.fixture(autouse=True, scope="class")
def results(sota_data_dir):
    yield
    if sota_data_dir:
        Tsc.write_common_metrics_file(per_model_metric_file_dump_path=pytest.metrics_dump_path)
        if Tsc.test == "eval":
            header = ["Model", "Metrics type", "Expected", "Measured", "Reference FP32", "Diff FP32", "Diff Expected",
                      "Error"]
        else:
            header = ["Model", "Metrics type", "Expected", "Measured", "Diff Expected", "Error"]
        Tsc().write_results_table(header, pytest.metrics_dump_path)
