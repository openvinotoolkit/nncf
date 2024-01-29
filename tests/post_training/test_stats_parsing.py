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


import datetime

from tests.post_training.pipelines.base import WCTimeStats

stdout = """
meta-llama/Llama-2-7b-chat-hf ----> int4_sym_g128_r80_data_awq
cache/llama-2-7b-chat-hf/fp16/openvino_model.xml
cache/llama-2-7b-chat-hf/int4_sym_g128_r80_data_awq/openvino_model.xml
compress weight arguments:  mode=CompressWeightsMode.INT4_SYM, ratio=0.8, group_size=128, all_layers=None, sensitivity_metric=None, awq=True
Statistics collection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 128/128 • 0:01:50 • 0:00:00
Searching for Mixed-Precision Configuration ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 224/224 • 0:06:08 • 0:00:00
Applying AWQ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 26/26 • 0:09:23 • 0:00:00
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 226/226 • 0:05:16 • 0:00:00
compressing weights took 1387.0 seconds
saving model llama-2-7b-chat-hf/int4_sym_g128_r80_data_awq/openvino_model.xml took 4.2 seconds
"""
def test_wc_parse():
    stats = WCTimeStats()
    stats.from_stdout(stdout)
    assert stats.result_dict() == {
        'Stat. collection time': datetime.datetime(1900, 1, 1, 0, 1, 50),
        'Mixed-Precision search time': datetime.datetime(1900, 1, 1, 0, 6, 8),
        'AWQ time': datetime.datetime(1900, 1, 1, 0, 9, 23),
        'Apply Compression time': datetime.datetime(1900, 1, 1, 0, 5, 16)
    }

