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

![alt text](<pics/absorbable lora adapters.png>)

- Tuning pipeline with distillation loss. The teacher model is the original bfloat16 model, while the student model
includes FQ operations. The training dataset is based on the training portion of the `wikitext-2-raw-v1` dataset,
consisting of 1024 samples of length 1024. Validation is performed at the end of each epoch using
[WhoWhatBench](https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/who_what_benchmark).
Training for 10 epochs on a single A100 GPU takes approximately 40 minutes for models with 1.7 billion parameters.
Alternatively, using three RTX 3090 GPUs, the process takes about 70 minutes.
The most significant accuracy improvements are usually observed within the first two epochs.

![alt text](pics/training_pipeline.png)

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

## Results

`HuggingFaceTB/SmolLM-1.7B-Instruct`

| Method                                | Main<br>Precision | Emb/Head<br>Precision | Group<br>Size | wikitext,<br>word_pp | WWB,<br>similarity |
|---------------------------------------|-------------------|-----------------------|---------------|----------------------|--------------------|
| Original   model (Torch)              | BF16              | BF16                  |               | 10.00                | 100%               |
| [QAT]   Mergeable LoRA                | INT4_ASYM         | INT8_ASYM             | 64            | 10.47                | 92%                |
| [PTQ]   AWQ + Scale Estimation + GPTQ | INT4_ASYM         | INT8_ASYM             | 64            | 10.71                | 91.2%              |
