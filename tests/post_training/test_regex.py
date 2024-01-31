import datetime

from tests.post_training.pipelines.lm_weight_compression import WCTimeStats

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

stdout = """
Model: tinyllama_data_aware
Backend: BackendType.OV
PTQ params: {'group_size': 64, 'ratio': 0.8, 'mode': <CompressWeightsMode.INT4_SYM: 'int4_sym'>}
Preparing...
Weight compression...
Statistics collection ━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 128/128 • 0:01:00 • 0:00:00
Searching for Mixed-Precision Configuration ━━━━━━ 100% 154/1… • 0:02:… • 0:00:…
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━ 100% 156/156 • 0:01:32 • 0:00:00
Validation...
Loading existing ground-truth validation data: /home/nlyaly/projects/nncf2/tests/post_training/fp32_models/tmp/tinyllama__tinyllama-1.1b-step-50k-105b/gold_all.csv
Evaluating of model in the directory: /home/nlyaly/projects/nncf2/tests/post_training/tmp/tinyllama_data_aware/OV

"""
def test_wc_parse():
    stats = WCTimeStats()
    stats.fill(stdout)
    assert stats.get_stats() == {
        'Stat. collection time': datetime.datetime(1900, 1, 1, 0, 1, 50),
        'Mixed-Precision search time': datetime.datetime(1900, 1, 1, 0, 6, 8),
        'AWQ time': datetime.datetime(1900, 1, 1, 0, 9, 23),
        'Apply Compression time': datetime.datetime(1900, 1, 1, 0, 5, 16)
    }
