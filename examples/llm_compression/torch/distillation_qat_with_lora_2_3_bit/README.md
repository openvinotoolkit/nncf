# Distillation QAT with absorbable LoRA for 2/3-bit LLM compression

This example trains a Hugging Face causal LM with NNCF weight compression in `FQ_LORA` mode. The script does not just run a static example; it performs an end-to-end compression + fine-tuning pipeline for 2-bit or 3-bit quantized weights.

## What the script actually does

Running `main.py` executes the following sequence:

1. Loads a pretrained model and tokenizer from Hugging Face.
2. Builds a training dataset from one of:
   - `pile-10k`
   - `wikitext-2`
   - `ultrachat_200k`
3. Optionally equalizes MLP weights before compression to improve low-bit robustness.
4. Applies `compress_weights(...)` with:
   - `INT3_SYM` for `--bits 3`
   - `INT2_SYM` for `--bits 2`
   - `compression_format=CompressionFormat.FQ_LORA`
5. Enables trainable absorbable LoRA adapters and quantizer scales.
6. Runs KL-based distillation training against the original model hidden states / logits.
7. Optionally performs a second scales-only fine-tuning phase.
8. Saves the NNCF checkpoint.
9. Exports the tuned model to OpenVINO IR.
10. Optionally exports a dequantized PyTorch model as well.

This is a real training pipeline for low-bit LLM optimization, not a simple inference example.

## Prerequisites

Before running this example, ensure you have following installed and set up your environment:

- Python 3.10+
- CUDA-enabled NVIDIA GPU
- PyTorch with CUDA support
- A working installation of the repository and dependencies

### 1. Create and activate a virtual environment

```bash
cd examples/llm_compression/torch/distillation_qat_with_lora_2_3_bit
python3 -m venv nncf_env
source nncf_env/bin/activate  # On Windows: nncf_env\Scripts\activate.bat
```

### 2. Install the example dependencies from this folder

```bash
python -m pip install -r requirements.txt
```

### 3. Install NNCF and other dependencies

```bash
python3 -m pip install ../../../../ -r requirements.txt
```

## Usage

Run the script directly:

```bash
cd examples/llm_compression/torch/distillation_qat_with_lora_2_3_bit
python main.py --pretrained meta-llama/Llama-3.2-1B-Instruct --bits 3 --dataset pile-10k --epochs 1 --output_dir output
```

The default command line is effectively:

```bash
python main.py \
  --pretrained meta-llama/Llama-3.2-1B-Instruct \
  --bits 3 \
  --output_dir output \
  --dataset pile-10k \
  --num_train_samples 2048 \
  --train_seqlen 1024 \
  --lr 1e-4 \
  --epochs 1 \
  --scale_epochs 0 \
  --batch_size 32 \
  --microbatch_size 8 \
  --lora_rank 256
```

## Important CLI options

### Model and compression

- `--pretrained`: HF model ID or local path.
- `--bits`: `2` or `3` bits per weight.
- `--lora_rank`: rank of absorbable LoRA adapters.
- `--basic_init`: use a simpler initialization without AWQ / scale estimation.
- `--equalize_mlp`: run MLP equalization before compression.
- `--resume`: continue from a saved checkpoint instead of reinitializing.

### Data

- `--dataset`: one of `pile-10k`, `wikitext-2`, or `ultrachat_200k`
- `--num_train_samples`: number of training samples
- `--train_seqlen`: context length for training samples

### Training

- `--lr`: base learning rate
- `--epochs`: number of main training epochs
- `--scale_epochs`: extra epochs where only quantizer scales are trained
- `--linear_lr_scheduler`: use a linear learning-rate decay schedule
- `--batch_size`: accumulation target
- `--microbatch_size`: per-step microbatch size

### Output/export

- `--output_dir`: root directory for logs, checkpoints, and exported models
- `--save_pt`: export a dequantized PyTorch model after training

## Output artifacts

The script writes artifacts under the chosen output directory. For example, with `--output_dir output` and `--bits 3`, it creates:

```text
output/
  tb/
    YYYY-MM-DD__HH-MM-SS/
  last_bits_3/
    nncf_checkpoint.pth
  last_bits_3/
    model.xml
    model.bin
    ...
```

The checkpoint contains the NNCF state and, when appropriate, the model state. A TensorBoard directory is created for loss and LR tracking. When `--save_pt` is enabled, the script also exports a dequantized PyTorch model under:

```text
output/last_bits_3/dequantized/
```

## Training behavior

The optimization loop is distillation-based:

- teacher hidden states are precomputed with the original model
- the compressed model receives the same tokens
- the script minimizes a KL divergence between the student and teacher outputs / hidden states
- optimizer updates are accumulated across microbatch steps and applied when `batch_size // microbatch_size` is reached

This makes the example a low-bit QAT + LoRA distillation workflow rather than a plain quantization pass.

## Notes

- The script asserts `torch.cuda.is_available()`, so it expects a CUDA-capable environment.
- Default model is `meta-llama/Llama-3.2-1B-Instruct`.
- OpenVINO export is always performed at the end of the run, after checkpoint restoration and stripping.
- `--resume` reuses the checkpoint if present; otherwise the script initializes from scratch.

For more background on absorbable LoRA and low-bit training-time compression, see the project documentation for QAT LoRA usage and NNCF compression flows.
