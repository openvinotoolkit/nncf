# Quantization-aware tuning with absorbable LoRA Adapters for improving accuracy of 4bit LLMs

This example demonstrates how to improve accuracy of Large Language Models (LLMs) with 4bit weights by
quantization-aware-training with absorbable LoRA adapters.

The example includes the following steps:

- Creation of NNCF model with extended FakeQuantize (FQ) operations on the weights of all linear layers,
except for the embedding and lm_head layers. This FQ includes absorbable LoRA Adapters and it performs fake quantization
in the following way: `dequantize(quantize(W + B @ A))`, where W is the original weight of the linear layer,
and A and B are the LoRA adapters. The compression part of the NNCF model is then saved in the NNCF checkpoint for
tuning and evaluation. It is expected that the initial accuracy of such a model is low, as it currently uses
a data-free Round-To-Nearest quantization scheme. In the next step, accuracy will be significantly improved by tuning
both the quantization scales and the LoRA adapters.

![alt text](/examples/llm_compression/torch/qat_with_lora/pics/absorbable_lora_adapters.png)

- Tuning pipeline with distillation loss. The teacher model is the original bfloat16 model, while the student model
includes FQ operations. The training dataset is based on the training portion of the `wikitext-2-raw-v1` dataset,
consisting of 1024 samples of length 1024. Validation is performed at the end of each epoch using
[WhoWhatBench](https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/who_what_benchmark).
Training for 10 epochs on a single A100 GPU takes approximately 40 minutes for models with 1.7 billion parameters.
Alternatively, using three RTX 3090 GPUs, the process takes about 70 minutes.
The most significant accuracy improvements are usually observed within the first two epochs.

![alt text](/examples/llm_compression/torch/qat_with_lora/pics/training_pipeline.png)

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
increase by 50% on average, achieving a **2x enhancement in minimizing accuracy loss**.

The **proportion of the PTWC-induced perplexity increase that is recovered** by using QAT+LoRA can be calculated
using the following formula:

$Improvement = \frac{PPL_{PTWC} - PPL_{QAT+LoRA}}{PPL_{PTWC} - PPL_{BF16}}$

Where:

- `PPL_BF16` is the perplexity of the original, uncompressed model (BF16 precision).
- `PPL_PTWC` is the perplexity after applying the best Post-Training Weight Compression method identified
for each specific model: this was "AWQ + Scale Estimation + GPTQ" for "HuggingFaceTB/SmolLM-1.7B-Instruct",
and "AWQ + Scale Estimation" for all other models evaluated.
- `PPL_QAT+LoRA` is the perplexity after applying Quantization-Aware Training with LoRA for 10 epochs.

All quantization methods compressed the models to `INT4_ASYM` precision with a group size of `64`.

| Model                               | Precision         | Wikitext,<br>word_ppl | Improvement |
|-------------------------------------|-------------------|-----------------------|-------------|
| google/gemma-2-2b-it                | BF16              | 15.02                 |             |
| google/gemma-2-2b-it                | INT4 (QAT + LoRA) | 15.09                 | 91%         |
| google/gemma-2-2b-it                | INT4 (best PTWC)  | 15.80                 |             |
| microsoft/phi3-mini-4k-instruct     | BF16              | 9.49                  |             |
| microsoft/phi3-mini-4k-instruct     | INT4 (QAT + LoRA) | 10.04                 | 37%         |
| microsoft/phi3-mini-4k-instruct     | INT4 (best PTWC)  | 10.36                 |             |
| Qwen/Qwen2.5-3B-Instruct            | BF16              | 11.01                 |             |
| Qwen/Qwen2.5-3B-Instruct            | INT4 (QAT + LoRA) | 11.44                 | 33%         |
| Qwen/Qwen2.5-3B-Instruct            | INT4 (best PTWC)  | 11.65                 |             |
| HuggingFaceTB/SmolLM-1.7B-Instruct  | BF16              | 19.11                 |             |
| HuggingFaceTB/SmolLM-1.7B-Instruct  | INT4 (QAT + LoRA) | 19.34                 | 66%         |
| HuggingFaceTB/SmolLM-1.7B-Instruct  | INT4 (best PTWC)  | 19.79                 |             |
| mistralai/Mistral-7B-v0.3           | BF16              | 8.21                  |             |
| mistralai/Mistral-7B-v0.3           | INT4 (QAT + LoRA) | 8.36                  | 20%         |
| mistralai/Mistral-7B-v0.3           | INT4 (best PTWC)  | 8.40                  |             |
| meta-llama/Llama-3.2-1B-Instruct    | BF16              | 16.30                 |             |
| meta-llama/Llama-3.2-1B-Instruct    | INT4 (QAT + LoRA) | 17.12                 | 40%         |
| meta-llama/Llama-3.2-1B-Instruct    | INT4 (best PTWC)  | 17.67                 |             |
| meta-llama/Llama-3.2-3B-Instruct    | BF16              | 12.67                 |             |
| meta-llama/Llama-3.2-3B-Instruct    | INT4 (QAT + LoRA) | 13.00                 | 39%         |
| meta-llama/Llama-3.2-3B-Instruct    | INT4 (best PTWC)  | 13.22                 |             |
| meta-llama/Meta-Llama-3-8B-Instruct | BF16              | 10.22                 |             |
| meta-llama/Meta-Llama-3-8B-Instruct | INT4 (QAT + LoRA) | 10.30                 | 62%         |
| meta-llama/Meta-Llama-3-8B-Instruct | INT4 (best PTWC)  | 10.45                 |             |
| microsoft/phi3.5-mini-instruct      | BF16              | 10.00                 |             |
| microsoft/phi3.5-mini-instruct      | INT4 (QAT + LoRA) | 10.53                 | 37%         |
| microsoft/phi3.5-mini-instruct      | INT4 (best PTWC)  | 10.71                 |             |
|                                     |                   |               Average | 46%         |
