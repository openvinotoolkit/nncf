# SQFT's NLS Tuning with Downstream Tasks 

<p align="center">
  <img src="/examples/llm_compression/torch/qat_with_lora/pics/lora_vs_nls.png" alt="alt text" width="400"/>
</p>

[main_nls.py](./main_nls.py) script supports fine-tuning and evaluating a language model with quantization-aware training and Neural Low-Rank Adapter Search (NLS) proposed by [Shears](https://arxiv.org/abs/2404.10934) and [SQFT](https://arxiv.org/abs/2410.03750) on various downstream tasks. For example, to run the script for the task `openbookqa`, you can use the following command:

```bash
python main_nls.py --pretrained Qwen/Qwen2.5-3B-Instruct --output_dir output --do_train --task openbookqa --lr 1e-4 --epochs 3 --batch_size 16 --eval_batch_size 64 --lora_rank_space 32 24 16
```

- `--pretrained`: The model ID or path of a pretrained Hugging Face model configuration.
- `--output_dir`: Path to the directory for storing logs, tuning checkpoints, compressed models, and evaluation results.
- `--do_train`: Whether to perform training. If not specified, the script will only evaluate the compressed model.
- `--task`: The evaluation task to be performed. Choices: ["gsm8k", "hellaswag", "openbookqa", "winogrande", "arc_challenge", "arc_easy"].
- `--lr`: Learning rate for fine-tuning. 
- `--epochs`: Number of epochs for training. 
- `--batch_size`: Size of the training batch. 
- `--microbatch_size`: Size of each training microbatch. Gradients will be accumulated until the batch size is reached.
- `--eval_batch_size`: Size of the batch for evaluation.
- `--lora_rank_space`: Specifies the search space for LoRA adapter ranks. For example, [32, 24, 16] indicates the ranks to be considered during NLS training and searching.

This script will automatically use a heuristic to obtain a good configuration for evaluation, but more powerful LoRA rank configurations can be optionally obtained through more advanced search algorithms. Here, we also support testing an arbitrary configuration for evaluation after training. The following command will load the trained checkpoint and test the specified LoRA rank configuration:

```bash
python main_nls.py --pretrained Qwen/Qwen2.5-3B-Instruct --output_dir output --resume --task openbookqa --lora_rank_space 32 24 16 --custom_rank_config 32 24 16 24 24 32 24 32 32 16 24 16 24 32 24 16 24 24 32 32 24 32 32 16 32 32 24 32
```

- `--resume`: Whether to resume training from a checkpoint. If specified, the script will load the trained checkpoint and continue training or evaluation.
- `--custom_rank_config`: Specifies the LoRA rank of adapters per layer.

## Results

The table illustrates that Quantization-Aware Training integrated with absorbable QAT + LoRA / QAT + NLS substantially improves the performance of compressed models on downstream tasks, and QAT + NLS performs better than QAT + LoRA overall.

The average score in the table represent the average accuracy of the four downstream tasks, `openbookqa`, `winogrande`, `arc_challenge` and `arc_easy` (all are "acc_norm" except `winogrande` which is "acc" of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)). For QAT + LoRA and QAT + NLS, we conducted experiments with epochs set to 3, 4, and 5, LoRA rank set to 16 and 32, the corresponding LoRA rank space of NLS set to `[16,12,8]` and `[32,24,16]`. We present the best results for each method. All quantization methods compressed the models to `INT4_ASYM` precision with a group size of `64`.


| Model                                | Precision          | Average score |
|--------------------------------------|--------------------|---------------|
| google/gemma-2-2b-it                 | BF16               | 0.6133        |
|                                      | INT4 (QAT + LoRA)  | 0.6801        |
|                                      | INT4 (QAT + NLS)   | **0.6843**    |
| Qwen/Qwen2.5-3B-Instruct             | BF16               | 0.5814        |
|                                      | INT4 (QAT + LoRA)  | 0.6916        |
|                                      | INT4 (QAT + NLS)   | **0.6966**    |
| mistralai/Mistral-7B-v0.3            | BF16               | 0.6209        |
|                                      | INT4 (QAT + LoRA)  | 0.7164        |
|                                      | INT4 (QAT + NLS)   | **0.7291**    |
| meta-llama/Llama-3.2-3B-Instruct     | BF16               | 0.5435        |
|                                      | INT4 (QAT + LoRA)  | 0.6510        |
|                                      | INT4 (QAT + NLS)   | **0.6570**    |
| HuggingFaceTB/SmolLM-1.7B-Instruct   | BF16               | 0.4934        |
|                                      | INT4 (QAT + LoRA)  | **0.5765**    |
|                                      | INT4 (QAT + NLS)   | 0.5733        |
| meta-llama/Meta-Llama-3-8B           | BF16               | 0.6233        |
|                                      | INT4 (QAT + LoRA)  | 0.7236        |
|                                      | INT4 (QAT + NLS)   | **0.7350**    |
| meta-llama/Meta-Llama-3-8B-Instruct  | BF16               | 0.6286        |
|                                      | INT4 (QAT + LoRA)  | 0.7076        |
|                                      | INT4 (QAT + NLS)   | **0.7128**    |
| meta-llama/Llama-3.1-8B              | BF16               | 0.6310        |
|                                      | INT4 (QAT + LoRA)  | 0.7243        |
|                                      | INT4 (QAT + NLS)   | **0.7297**    |
| meta-llama/Llama-3.1-8B-Instruct     | BF16               | 0.6297        |
|                                      | INT4 (QAT + LoRA)  | 0.7140        |
|                                      | INT4 (QAT + NLS)   | **0.7166**    |
| Qwen/Qwen2.5-7B                      | BF16               | 0.6207        |
|                                      | INT4 (QAT + LoRA)  | 0.7366        |
|                                      | INT4 (QAT + NLS)   | **0.7408**    |
| Qwen/Qwen2.5-7B-Instruct             | BF16               | 0.6401        |
|                                      | INT4 (QAT + LoRA)  | 0.7356        |
|                                      | INT4 (QAT + NLS)   | **0.7382**    |


## Citation
If you find this code and the NLS technique helpful, please kindly cite:
```bibtex
@inproceedings{munoz2025lowrank,
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
@inproceedings{munoz-etal-2024-sqft,
    title = "{SQFT}: Low-cost Model Adaptation in Low-precision Sparse Foundation Models",
    author = "Munoz, Juan Pablo  and
      Yuan, Jinjie  and
      Jain, Nilesh",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.749",
    pages = "12817--12832",
}
```
