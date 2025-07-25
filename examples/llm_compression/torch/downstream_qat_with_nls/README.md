# Quantization-aware NLS Tuning for improving accuracy on downstream Tasks

This example demonstrates how to improve accuracy of Large Language Models (LLMs) with 4-bit weights by
quantization-aware training with **Neural Low-Rank Adapter Search (NLS)** on downstream tasks.
It uses **absorbable LoRA adapters** and the **`FQ_LORA_NLS`** compression format.
For detailed information about the methodology and format, please refer to this [page](../../../../docs/usage/training_time_compression/quantization_aware_training_lora/Usage.md).

<p align="center">
  <img src="/examples/llm_compression/torch/downstream_qat_with_nls/pics/lora_vs_nls.png" alt="LoRA vs NLS" width="400"/>
</p>

## Prerequisites

Before running this example, ensure you have Python 3.9+ installed and set up your environment:

### 1. Create and activate a virtual environment

```bash
python3 -m venv nncf_env
source nncf_env/bin/activate  # On Windows: nncf_env\Scripts\activate.bat
```

### 2. Install NNCF and other dependencies

```bash
python3 -m pip install ../../../../ -r requirements.txt
```

## Run Example

[main.py](main.py) supports fine-tuning and evaluating a language model with quantization-aware training and **Neural Low-Rank Adapter Search (NLS)** proposed by [Shears](https://arxiv.org/abs/2404.10934) and [SQFT](https://arxiv.org/abs/2410.03750) on various downstream tasks. For example, to run the script for the task [openbookqa](https://huggingface.co/datasets/allenai/openbookqa), you can use the following command:

```bash
python main.py --pretrained Qwen/Qwen2.5-3B-Instruct --output_dir output --fast_eval --task openbookqa --lr 1e-4 --epochs 3 --batch_size 16 --eval_batch_size 64 --lora_rank_space 32 24 16
```

- `--pretrained`: The model ID or path of a pretrained Hugging Face model configuration.
- `--output_dir`: Path to the directory for storing logs, tuning checkpoints, compressed models, and evaluation results.
- `--fast_eval`: Enable faster evaluation by applying in-place quantization to the model weights.
- `--task`: The evaluation task to be performed. Choices: ["gsm8k", "hellaswag", "openbookqa", "winogrande", "arc_challenge", "arc_easy"].
- `--lr`: Learning rate for fine-tuning.
- `--epochs`: Number of epochs for training.
- `--batch_size`: Size of the training batch.
- `--eval_batch_size`: Size of the batch for evaluation.
- `--lora_rank_space`: Specifies the search space for LoRA adapter ranks. For example, [32, 24, 16] indicates the ranks to be considered during NLS training and searching.
- `--eval_only`: Whether to perform evaluation only. If specified, the model will be loaded from the checkpoint for evaluation.
- `--resume`: Whether to resume training from a checkpoint. If specified, the script will load the trained checkpoint and continue training or evaluation.
- `--custom_rank_config`: Specifies the LoRA rank of adapters per layer.
- `--num_min_loss_configs`: Number of configurations to evaluate for the min loss heuristic.

Regarding evaluation, the script will automatically use a heuristic to obtain a good configuration for evaluation. This default strategy takes advantage of some information from the training phase and requires the evaluation of only 7 suggested configurations (median + frequent + 5 min loss). This is automatically done in the example script, and only the best configuration from these candidates is returned to the user. More powerful elastic LoRA NLS configurations can be optionally obtained through more advanced search algorithms. We also support testing a custom configuration for evaluation after training. The following command will load the trained checkpoint and test the specified LoRA rank configuration:

```bash
python main.py --pretrained Qwen/Qwen2.5-3B-Instruct --output_dir output --fast_eval --resume --eval_only --task openbookqa --lora_rank_space 32 24 16 --custom_rank_config 32 24 16 24 24 32 24 32 32 16 24 16 24 32 24 16 24 24 32 32 24 32 32 16 32 32 24 32
```

This script also supports running the vanilla LoRA method. We only need to pass a single number for `--lora_rank_space`, such as `--lora_rank_space 32`. In addition, the training time of LoRA and NLS is very similar, and there is almost no overhead in activating different sub-adapters during training. For instance, fine-tuning the compressed Llama-3.2-3B-Instruct model for 3 epochs on [arc-challenge](https://huggingface.co/datasets/allenai/ai2_arc) takes 161.83 seconds with LoRA and 164.89 seconds with NLS.

## Results

The table below illustrates the performance improvements achieved by integrating Quantization-Aware Training (QAT) with absorbable LoRA and NLS on compressed models across various downstream tasks before exporting to OpenVINO. Our evaluation encompasses 11 large language models and 4 downstream tasks: [openbookqa](https://huggingface.co/datasets/allenai/openbookqa), [winogrande](https://huggingface.co/datasets/allenai/winogrande), [arc-challenge](https://huggingface.co/datasets/allenai/ai2_arc), and [arc-easy](https://huggingface.co/datasets/allenai/ai2_arc). The value in the table represents the mean accuracy of these tasks, with "acc_norm" used for all except winogrande, which uses "acc" ([lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)).

To ensure a fair and comprehensive comparison, we conducted experiments with epochs set to 3, 4, and 5, LoRA rank set to 16 and 32, and the corresponding LoRA rank space of NLS set to `[16,12,8]` and `[32,24,16]`. We present the best results.
INT4 (LoRA + PTWC) results are derived from the best BF16 (LoRA) model using the OpenVINO PTWC (AWQ + Scale Estimation + GPTQ) method. All quantization methods compressed the models to `INT4_ASYM` precision with a group size of 64. For BF16 model + LoRA finetuning, we used [PEFT](https://github.com/huggingface/peft) for inserting the LoRA adapters, ensuring consistent adapter placement with QAT.

**Conclusion:** The results indicate a performance comparison among the methods:
**BF16 < INT4 (LoRA + PTWC) < INT4 (QAT + LoRA) < INT4 (QAT + NLS) ≲ BF16 (LoRA)**. This demonstrates that QAT + NLS generally provides the best performance among the quantized models, closely approaching the performance of BF16 (LoRA).

\* We highlight the best of the INT4 results.

| Model                                | BF16  | BF16 (LoRA) | INT4 (LoRA + PTWC) | INT4 (QAT + LoRA) | INT4 (QAT + NLS) |
|--------------------------------------|-------|-------------|--------------------|-------------------|------------------|
| meta-llama/Meta-Llama-3-8B           | 0.6233| 0.7277      | 0.7167             | 0.7286            | **0.7344**       |
| meta-llama/Meta-Llama-3-8B-Instruct  | 0.6286| 0.7148      | 0.7098             | 0.7116            | **0.7160**       |
| meta-llama/Llama-3.1-8B              | 0.6310| 0.7330      | 0.7201             | 0.7216            | **0.7306**       |
| meta-llama/Llama-3.1-8B-Instruct     | 0.6297| 0.7197      | 0.7160             | 0.7152            | **0.7183**       |
| Qwen/Qwen2.5-7B                      | 0.6207| 0.7344      | 0.7269             | 0.7317            | **0.7369**       |
| Qwen/Qwen2.5-7B-Instruct             | 0.6401| 0.7305      | 0.7234             | 0.7301            | **0.7380**       |
| mistralai/Mistral-7B-v0.3            | 0.6209| 0.7208      | 0.7115             | 0.7219            | **0.7246**       |
| Qwen/Qwen2.5-3B-Instruct             | 0.5814| 0.7003      | 0.6839             | 0.6914            | **0.6940**       |
| meta-llama/Llama-3.2-3B-Instruct     | 0.5435| 0.6515      | 0.6503             | 0.6564            | **0.6612**       |
| HuggingFaceTB/SmolLM-1.7B-Instruct   | 0.4934| 0.5759      | **0.5751**         | 0.5714            | 0.5695           |
| google/gemma-2-2b-it                 | 0.6133| 0.6806      | 0.6658             | 0.6721            | **0.6768**       |

## Citation

If you find this code and the NLS technique helpful, please kindly cite:

```bibtex
@inproceedings{munoz2025low,
    title=Low-Rank Adapters Meet Neural Architecture Search for LLM Compression,
    author="Munoz, J. Pablo  and
      Yuan, Jinjie  and
      Jain, Nilesh",,
    booktitle={AAAI'25 workshop on CoLoRAI - Connecting Low-Rank Representations in AI},
    year={2025},
    url={https://arxiv.org/abs/2501.16372}
}
```

```bibtex
@inproceedings{munoz-2024-sqft,
    title = "{SQFT}: Low-cost Model Adaptation in Low-precision Sparse Foundation Models",
    author = "Munoz, Juan Pablo  and
      Yuan, Jinjie  and
      Jain, Nilesh",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.749",
    pages = "12817--12832",
}
```
