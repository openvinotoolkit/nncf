import os
from pathlib import Path
from typing import List
import traceback
from nncf import compress_weights
from tqdm import trange
import time
import json
import shutil
import random
import copy
from optimum.intel.openvino import (
    OVModelForCausalLM,
    # OVQwenModpip iel
)
from transformers import (
    AutoTokenizer,
    AutoConfig
)
from contextlib import redirect_stdout, redirect_stderr
from nncf.parameters import CompressWeightsMode
from whowhatbench import Evaluator

from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
from optimum.exporters import TasksManager
from openvino import Core
core = Core()
from datasets import load_dataset


TasksManager._SUPPORTED_MODEL_TYPE["stablelm-epoch"] = TasksManager._SUPPORTED_MODEL_TYPE["llama"]
NormalizedConfigManager._conf["stablelm-epoch"] = NormalizedTextConfig.with_args(
    num_layers="num_hidden_layers",
    num_attention_heads="num_attention_heads",
)
NormalizedConfigManager._conf['qwen'] = NormalizedTextConfig.with_args(
    num_layers='num_hidden_layers', num_attention_heads='num_attention_heads', hidden_size='hidden_size'
)

MODEL_IDS = [
    # 'stable-zephyr-3b-dpo',
    # 'llama-2-7b-chat-hf',
    # 'stablelm-3b-4e1t',
    # 'zephyr-7b-beta',
    # 'qwen-7b-chat',
    # 'facebook/opt-125m',
    'TinyLlama/TinyLlama-1.1B-step-50K-105b',
]
for MODEL_ID in MODEL_IDS:
    start_time = time.time()

    model_name = Path(MODEL_ID).name.lower()
    cache_dir = Path('cache')
    if cache_dir.is_symlink():
        cache_dir = cache_dir.readlink()

    ROOT_DIR = cache_dir / model_name
    gold_folder = ROOT_DIR / "fp16"
    gold_ir_path = gold_folder / "openvino_model.xml"
    gold_csv = gold_folder / 'gold_all.csv'
    # config_gold = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer_gold = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer_gold.save_pretrained(gold_folder)
    print('gold path:', gold_csv.resolve())
    if gold_csv.exists():
        evaluator = Evaluator(tokenizer=tokenizer_gold, gt_data=gold_csv, test_data=str(gold_csv))
    else:
        model_gold = OVModelForCausalLM.from_pretrained(
            gold_folder,
            # config=config_gold,
            trust_remote_code=True,
            use_cache=True,
            load_in_8bit=False,
            compile=False,
            stateful=False
        )
        evaluator = Evaluator(base_model=model_gold, tokenizer=tokenizer_gold)
        evaluator.dump_gt(str(gold_csv))

    print('Load gold time: {:.2f} seconds'.format(time.time() - start_time))

    EXP_NAMES = [
        'int4_sym_g64_r80',
        'int4_sym_g64_r80_data',
        # 'int4_sym_g64_r80_data_awq',
    ]

    for exp_name in EXP_NAMES:
        start_time = time.time()
        cmp_ir_folder = ROOT_DIR / exp_name
        cmp_ir_path = cmp_ir_folder / "openvino_model.xml"
        # cmp_model = core.read_model(model=cmp_ir_path)
        cmp_model = OVModelForCausalLM.from_pretrained(
            cmp_ir_folder,
            # config=config_gold,
            trust_remote_code=True,
            use_cache=True
        )
        all_metrics_per_question, all_metrics = evaluator.score(cmp_model)
        print(all_metrics)
        similarity = float(all_metrics['similarity'].iloc[0])
        sdt_norm = float(all_metrics['SDT norm'].iloc[0])
        score = (similarity + 1 - sdt_norm) / 2
        # print(all_metrics_per_question)
        print('final score=', score)
        print('Evaluate time: {:.2f} seconds'.format(time.time() - start_time))
        all_metrics['weighted'] = [score]
        model_cache_dir = cmp_ir_folder / 'model_cache'
        if model_cache_dir.exists():
            shutil.rmtree(model_cache_dir)
        all_metrics.to_csv(cmp_ir_folder / 'eval.csv')