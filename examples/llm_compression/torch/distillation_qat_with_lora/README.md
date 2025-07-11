# Knowledge Distillation with absorbable LoRA Adapters for general accuracy improvement of 4-bit LLMs

<p align="center">
    <img src="/examples/llm_compression/torch/distillation_qat_with_lora/pics/training_pipeline.png" alt="distillation" width="600"/>
</p>

## Overview

This example demonstrates how to enhance the accuracy of Large Language Models (LLMs) with 4-bit weights.
It uses quantization-aware training (QAT) with **absorbable LoRA adapters** and the **`FQ_LORA`** compression format.
The method employs traditional low-rank adaptation (LoRA) with fixed ranks across layers and knowledge distillation,
where an uncompressed model teaches a compressed student model without task-specific over-fitting.

For detailed information about the methodology and format, please refer to this [page](../../../../docs/usage/training_time_compression/quantization_aware_training_lora/Usage.md).

## Install requirements

To use this example:

- Create a separate Python* environment and activate it: `python3 -m venv nncf_env && source nncf_env/bin/activate`
- Install dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
pip install -e ../../../../
```

## Run Example

The example is fully automated. Just run the following command in the prepared Python environment:

```bash
python main.py
```

## Results on wikitext

The table illustrates that Quantization-Aware Training integrated with absorbable LoRA (QAT+LoRA) substantially
alleviates the accuracy degradation associated with quantization. Compared to NNCF's prior leading
post-training weight compression (PTWC) techniques, QAT+LoRA reduces the quantization-induced perplexity
increase by around 50% on average, achieving a **2x enhancement in minimizing accuracy loss**.

The **proportion of the PTWC-induced perplexity increase that is recovered** by using QAT+LoRA can be calculated
using the following formula:

$Improvement = \frac{PPL_{PTWC} - PPL_{QAT+LoRA}}{PPL_{PTWC} - PPL_{BF16}}$

Where:

- `PPL_BF16` is the perplexity of the original, uncompressed model (BF16 precision).
- `PPL_PTWC` is the perplexity after applying the best Post-Training Weight Compression method for each model
(selected from either "AWQ + Scale Estimation + GPTQ" or "AWQ + Scale Estimation").
- `PPL_QAT+LoRA` is the perplexity after applying Quantization-Aware Training with LoRA adapters for 10 epochs.

The training dataset comprises 1024 samples (each 1024 tokens long) from the training split of the `wikitext-2-raw-v1` dataset. Validation occurs after each epoch using [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) on the validation split of the same dataset.
Final perplexity measurements are conducted using OpenVINO with dynamic quantization and float16 KV-cache enabled on the test split of `wikitext-2-raw-v1`.
For `HuggingFaceTB/SmolLM-1.7B-Instruct` model training with evaluation for 10 epochs requires about 25 minutes on a single A100 GPU or approximately 50 minutes using three RTX 3090 GPUs.

All quantization methods compressed the models to `INT4_ASYM` precision with a group size of `64`.

| Model                               | Precision         | Wikitext,<br>word_ppl | Improvement |
|-------------------------------------|-------------------|-----------------------|-------------|
| google/gemma-2-2b-it                | BF16              | 15.05                 |             |
| google/gemma-2-2b-it                | INT4 (QAT + LoRA) | 15.28                 | 69%         |
| google/gemma-2-2b-it                | INT4 (best PTWC)  | 15.80                 |             |
| HuggingFaceTB/SmolLM-1.7B-Instruct  | BF16              | 19.11                 |             |
| HuggingFaceTB/SmolLM-1.7B-Instruct  | INT4 (QAT + LoRA) | 19.57                 | 30%         |
| HuggingFaceTB/SmolLM-1.7B-Instruct  | INT4 (best PTWC)  | 19.77                 |             |
| meta-llama/Llama-3.2-1B-Instruct    | BF16              | 16.29                 |             |
| meta-llama/Llama-3.2-1B-Instruct    | INT4 (QAT + LoRA) | 17.02                 | 40%         |
| meta-llama/Llama-3.2-1B-Instruct    | INT4 (best PTWC)  | 17.51                 |             |
| meta-llama/Llama-3.2-3B-Instruct    | BF16              | 12.67                 |             |
| meta-llama/Llama-3.2-3B-Instruct    | INT4 (QAT + LoRA) | 13.00                 | 40%         |
| meta-llama/Llama-3.2-3B-Instruct    | INT4 (best PTWC)  | 13.22                 |             |
| meta-llama/Meta-Llama-3-8B-Instruct | BF16              | 10.20                 |             |
| meta-llama/Meta-Llama-3-8B-Instruct | INT4 (QAT + LoRA) | 10.34                 | 44%         |
| meta-llama/Meta-Llama-3-8B-Instruct | INT4 (best PTWC)  | 10.45                 |             |
| microsoft/phi3.5-mini-instruct      | BF16              | 9.98                  |             |
| microsoft/phi3.5-mini-instruct      | INT4 (QAT + LoRA) | 10.46                 | 34%         |
| microsoft/phi3.5-mini-instruct      | INT4 (best PTWC)  | 10.71                 |             |
| microsoft/phi3-mini-4k-instruct     | BF16              | 9.48                  |             |
| microsoft/phi3-mini-4k-instruct     | INT4 (QAT + LoRA) | 10.03                 | 28%         |
| microsoft/phi3-mini-4k-instruct     | INT4 (best PTWC)  | 10.24                 |             |
| mistralai/Mistral-7B-v0.3           | BF16              | 8.21                  |             |
| mistralai/Mistral-7B-v0.3           | INT4 (QAT + LoRA) | 8.35                  | 23%         |
| mistralai/Mistral-7B-v0.3           | INT4 (best PTWC)  | 8.40                  |             |
| Qwen/Qwen2.5-3B-Instruct            | BF16              | 11.02                 |             |
| Qwen/Qwen2.5-3B-Instruct            | INT4 (QAT + LoRA) | 11.48                 | 27%         |
| Qwen/Qwen2.5-3B-Instruct            | INT4 (best PTWC)  | 11.64                 |             |
|                                     |                   |               Average | 39%         |
