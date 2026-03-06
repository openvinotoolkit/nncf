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

from re import compile
from subprocess import check_output  # nosec B404: used only for executing benchmark_app

throughput_pattern = compile(r"Throughput\: (.+?) FPS")


def execute_benchmark_on_cpu(model_path, time, shape=None):
    command = ["benchmark_app", "-m", model_path.as_posix(), "-d", "CPU", "-api", "async", "-t", str(time)]
    if shape is not None:
        command += ["-shape", str(shape)]

    cmd_output = check_output(command, text=True)  # nosec B603: used only for executing benchmark_app
    print(*cmd_output.splitlines()[-8:], sep="\n")

    match = throughput_pattern.search(cmd_output)
    return float(match.group(1))
