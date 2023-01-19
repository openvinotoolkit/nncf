"""
 Copyright (c) 2023 Intel Corporation
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

from typing import List
import csv
import re

from tests.shared.command import Command


def run_command(command: List[str]):
    com_str = ' '.join(command)
    print(f"Run command: {com_str}")
    runner = Command(com_str)
    runner.run()
    cmd_output = " ".join(runner.output)
    return cmd_output


def download_model(name, path):
    com_line = [
        'omz_downloader',
        "--name", name,
        "-o", str(path)
    ]
    cmd_output = run_command(com_line)
    re_exp = r"========== (Downloading|Retrieving) ([^\s]+)"
    match = re.search(re_exp, str(cmd_output))
    model_path = match.group(2)
    return model_path


def convert_model(name, path, model_precision='FP32'):
    com_line = [
        'omz_converter',
        "--name", name,
        "-d", str(path),
        "--precisions", model_precision,
        "-o", str(path)
    ]
    cmd_output = run_command(com_line)

    re_exp = r"XML file: ([^\s]+)"
    match = re.search(re_exp, str(cmd_output))
    model_path = match.group(1)
    return model_path


def get_metrics(ac_report):
    metrics = {}
    with open(ac_report, 'r', encoding='utf8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metric_name = row['metric_name']
            metric_value = row['metric_value']
            metrics[metric_name] = metric_value
    return metrics
