import os
import shutil
import subprocess
import threading
import time
from pathlib import Path


def stream_handler(stream, target_file):
    for line in iter(stream.readline, ''):
        print(line, end='')
        target_file.write(line)


parent_model_dir = Path("/home/nsavel/workspace/openvino.genai/llm_bench/python/models")
parent_log_dir = Path("compression_logs")

experiment_params = [
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--numpy"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --recompile"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --release-memory"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --recompile --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --release-memory --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic --recompile"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic --release-memory"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic --recompile --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic --release-memory --share-outputs"),
    
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--numpy"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --recompile"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --release-memory"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --recompile --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --release-memory --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic --recompile"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic --release-memory"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic --recompile --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic --release-memory --share-outputs"),
    
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--numpy"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --recompile"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --release-memory"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --recompile --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --release-memory --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic --recompile"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic --release-memory"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic --recompile --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/tiny-llama", "--end-to-end --dynamic --release-memory --share-outputs"),


    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--numpy"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --recompile"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --release-memory"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --recompile --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --release-memory --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic --recompile"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic --release-memory"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic --recompile --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic --release-memory --share-outputs"),
    
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--numpy"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --recompile"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --release-memory"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --recompile --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --release-memory --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic --recompile"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic --release-memory"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic --recompile --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic --release-memory --share-outputs"),
    
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--numpy"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --recompile"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --release-memory"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --recompile --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --release-memory --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic --recompile"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic --release-memory"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic --recompile --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/phi3", "--end-to-end --dynamic --release-memory --share-outputs"),


    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--numpy"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --recompile"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --release-memory"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --recompile --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --release-memory --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic --recompile"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic --release-memory"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic --recompile --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic --release-memory --share-outputs"),

    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--numpy"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --recompile"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --release-memory"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --recompile --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --release-memory --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic --recompile"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic --release-memory"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic --recompile --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic --release-memory --share-outputs"),

    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--numpy"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --recompile"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --release-memory"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --recompile --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --release-memory --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic --recompile"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic --release-memory"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic --recompile --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int8/llama3-8b", "--end-to-end --dynamic --release-memory --share-outputs"),
    



    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --numpy"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym "),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --recompile"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --release-memory"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --recompile --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --release-memory --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end --recompile"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end --release-memory"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end --recompile --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end --release-memory --share-outputs"),

    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --numpy"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym "),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --recompile"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --release-memory"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --recompile --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --release-memory --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end --recompile"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end --release-memory"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end --recompile --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end --release-memory --share-outputs"),

    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --numpy"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --recompile"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --release-memory"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --recompile --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --release-memory --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end --recompile"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end --release-memory"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end --recompile --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/tiny-llama", "--compression-mode int4_asym --end-to-end --release-memory --share-outputs"),


    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --numpy"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym "),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --recompile"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --release-memory"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --recompile --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --release-memory --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end --recompile"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end --release-memory"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end --recompile --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end --release-memory --share-outputs"),

    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --numpy"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym "),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --recompile"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --release-memory"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --recompile --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --release-memory --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end --recompile"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end --release-memory"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end --recompile --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end --release-memory --share-outputs"),

    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --numpy"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --recompile"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --release-memory"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --recompile --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --release-memory --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end --recompile"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end --release-memory"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end --recompile --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/phi3", "--compression-mode int4_asym --end-to-end --release-memory --share-outputs"),


    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --numpy"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym "),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --recompile"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --release-memory"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --recompile --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --release-memory --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --recompile"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --release-memory"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --recompile --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP32", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --release-memory --share-outputs"),

    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --numpy"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym "),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --recompile"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --release-memory"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --recompile --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --release-memory --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --recompile"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --release-memory"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --recompile --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/FP16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --release-memory --share-outputs"),

    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --numpy"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --recompile"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --release-memory"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --recompile --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --release-memory --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --recompile"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --release-memory"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --recompile --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/BF16", parent_log_dir / "recompile_vs_release-memory/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --release-memory --share-outputs"),



    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "optimal_configurations/int8/tiny-llama", "--save-model --numpy"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "optimal_configurations/int8/tiny-llama", "--save-model --end-to-end --release-memory"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "optimal_configurations/int8/tiny-llama", "--save-model --numpy"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "optimal_configurations/int8/tiny-llama", "--save-model --end-to-end --release-memory"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "optimal_configurations/int8/tiny-llama", "--save-model --numpy"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "optimal_configurations/int8/tiny-llama", "--save-model --end-to-end --release-memory"),

    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "optimal_configurations/int8/phi3", "--numpy"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "optimal_configurations/int8/phi3", "--end-to-end --release-memory"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "optimal_configurations/int8/phi3", "--numpy"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "optimal_configurations/int8/phi3", "--end-to-end --release-memory"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "optimal_configurations/int8/phi3", "--numpy"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "optimal_configurations/int8/phi3", "--end-to-end --release-memory"),

    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/optimum-fp32", parent_log_dir / "optimal_configurations/int8/llama3-8b", "--numpy"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/optimum-fp32", parent_log_dir / "optimal_configurations/int8/llama3-8b", "--end-to-end --release-memory"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/optimum-fp16", parent_log_dir / "optimal_configurations/int8/llama3-8b", "--numpy"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/optimum-fp16", parent_log_dir / "optimal_configurations/int8/llama3-8b", "--end-to-end --release-memory"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/optimum-bf16", parent_log_dir / "optimal_configurations/int8/llama3-8b", "--numpy"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/optimum-bf16", parent_log_dir / "optimal_configurations/int8/llama3-8b", "--end-to-end --release-memory"),
    
    

    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "optimal_configurations/int4/tiny-llama", "--save-model --compression-mode int4_asym --numpy"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP32", parent_log_dir / "optimal_configurations/int4/tiny-llama", "--save-model --compression-mode int4_asym --end-to-end"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "optimal_configurations/int4/tiny-llama", "--save-model --compression-mode int4_asym --numpy"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/FP16", parent_log_dir / "optimal_configurations/int4/tiny-llama", "--save-model --compression-mode int4_asym --end-to-end --release-memory --share-outputs"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "optimal_configurations/int4/tiny-llama", "--save-model --compression-mode int4_asym --numpy"),
    (parent_model_dir / "tiny-llama/pytorch/dldt/BF16", parent_log_dir / "optimal_configurations/int4/tiny-llama", "--save-model --compression-mode int4_asym --end-to-end --release-memory --share-outputs"),

    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "optimal_configurations/int4/phi3", "--compression-mode int4_asym --numpy"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP32", parent_log_dir / "optimal_configurations/int4/phi3", "--compression-mode int4_asym --end-to-end"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "optimal_configurations/int4/phi3", "--compression-mode int4_asym --numpy"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/FP16", parent_log_dir / "optimal_configurations/int4/phi3", "--compression-mode int4_asym --end-to-end --release-memory --share-outputs"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "optimal_configurations/int4/phi3", "--compression-mode int4_asym --numpy"),
    (parent_model_dir / "phi3-mini-4k-instruct/pytorch/dldt/BF16", parent_log_dir / "optimal_configurations/int4/phi3", "--compression-mode int4_asym --end-to-end --release-memory --share-outputs"),

    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/optimum-fp32", parent_log_dir / "optimal_configurations/int4/llama3-8b", "--compression-mode int4_asym --numpy"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/optimum-fp32", parent_log_dir / "optimal_configurations/int4/llama3-8b", "--compression-mode int4_asym --end-to-end"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/optimum-fp16", parent_log_dir / "optimal_configurations/int4/llama3-8b", "--compression-mode int4_asym --numpy"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/optimum-fp16", parent_log_dir / "optimal_configurations/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --release-memory --share-outputs"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/optimum-bf16", parent_log_dir / "optimal_configurations/int4/llama3-8b", "--compression-mode int4_asym --numpy"),
    (parent_model_dir / "Meta-Llama-3-8B/pytorch/dldt/optimum-bf16", parent_log_dir / "optimal_configurations/int4/llama3-8b", "--compression-mode int4_asym --end-to-end --release-memory --share-outputs"),
]

for model_dir, log_dir, params in experiment_params:
    model_path = model_dir / "openvino_model.xml"
    cmd = f"/home/nsavel/venvs/nncf/bin/python weight_compression.py --model-path {model_path} --log-dir {log_dir} {params}"

    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "log.txt", "a") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True,
            preexec_fn=os.setsid,
        )

        stdout_thread = threading.Thread(target=stream_handler, args=(process.stdout, log_file))
        stderr_thread = threading.Thread(target=stream_handler, args=(process.stderr, log_file))

        stdout_thread.start()
        stderr_thread.start()

        stdout_thread.join()
        stderr_thread.join()

        process.wait()
    time.sleep(10)

evaluated_paths = set()
for _, log_dir, _ in experiment_params:
    for model_path in log_dir.rglob("**/*"):
        model_path: Path
        if model_path.suffix != ".xml":
            continue
        if model_path.absolute() in evaluated_paths:
            continue
        evaluated_paths.add(model_path.absolute())

        model_dir = model_path.parent.absolute()
        cmd = f"/home/nsavel/venvs/lm-evaluation-harness/bin/lm_eval --model openvino --model_args pretrained={model_dir},trust_remote_code=True --tasks wikitext --output_path {model_dir}"
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
