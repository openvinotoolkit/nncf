# Integrating NNCF into Transformers
https://github.com/huggingface/transformers

This folder contains a git patch to enable NNCF-based quantization for XNLI, SQuAD and GLUE training pipelines of the huggingface transformers repository. 

Instructions:
1. Apply the `0001-Modifications-for-NNCF-usage.patch` file to the huggingface transformers repository checked out at commit id: `bff1c71e84e392af9625c345f9ea71f7b6d75fb3`

2. Install the `transformers` library and the example scripts from the patched repository as described in the documentation for the huggingface transformers repository.

3. To start quantization-aware fine-tuning of NLP models using NNCF, use the regular scripts and command line parameters for XNLI and SQuAD training, but with additional `--nncf_config <path_to_nncf_config>` parameter.
The NNCF configs to be used in this way are also provided in the same patch on a per-model, per-compression algorithm basis.
Distributed multiprocessing is also supported, simply use the corresponding version of the command line in the huggingface transformers repository with the same additional `--nncf_config` parameter.



4. While running with the `--nncf_config` option, the training scripts will output NNCF-wrapped model checkpoints instead of the regular ones. You may evaluate these checkpoints using the same command lines for training above, but with the`--do_train` key omitted. In order to export these checkpoints into ONNX format, further add `--to_onnx <path_to_output_onnx_file>` to your evaluation command line parameters.
See exact command lines for each case in the model notes below.
Note that in all cases the training hyperparameters might have to be adjusted to accomodate the hardware you have available.

## Current best results:

All models use as their baselines the checkpoints obtained with the scripts and command line parameters from the corresponding sections in the original repository documentation. While fine-tuning the quantized model, the hyperparameters were left unchanged, i.e. the difference in the training script invocation was limited to adding `--nncf_config` option and specifying the pre-trained baseline model as the starting point for quantization fine-tuning. For RoBERTa-MNLI, no baseline model finetuning was necessary since the `roberta-large-mnli` model pretrained on MNLI was already available for download.

Make sure that you are running evaluation on a single GPU, since the repository evaluation scripts give inconsistent results when running multi-GPU evaluation.

### BERT-XNLI

_Full-precision FP32 baseline model_ - bert-base-chinese, trained on the Chinese portion of XNLI - 77.68% accuracy when evaluated on the Chinese portion of XNLI test set.

_INT8 model (symmetric weights, asymmetric activations quantization)_ - 77.22% accuracy in the same evaluation conditions.

**INT8 model quantization-aware training command line:**

`python examples/pytorch/text-classification/run_xnli.py --model_name_or_path bert-base-chinese --language zh --train_language zh --do_train --do_eval --per_gpu_train_batch_size 48 --per_gpu_eval_batch_size 1 --learning_rate 5e-5 --num_train_epochs 4.0 --max_seq_length 128 --output_dir bert_xnli_int8 --save_steps 200 --nncf_config nncf_bert_config_xnli.json`

**Fine-tuned INT8 model evaluation and ONNX export command line:**

`python examples/pytorch/text-classification/run_xnli.py --model_name_or_path bert_xnli_int8 --language zh --train_language zh --do_eval --per_gpu_eval_batch_size 1 --max_seq_length 128 --output_dir bert_xnli_int8 --nncf_config nncf_bert_config_xnli.json --to_onnx bert_xnli_int8.onnx`


### BERT-SQuAD v1.1

_Full-precision FP32 baseline model_ - bert-large-uncased-whole-word-masking model, trained on SQuAD v1.1 - 93.21% F1, 87.2% EM on the dev set,

_INT8 model (symmetric quantization)_ - 92.55% F1, 86.1% EM on the dev set.

**INT8 model quantization-aware training command line (trained on 4x Tesla V100):**

`python examples/pytorch/question-answering/run_qa.py --model_name_or_path bert-large-uncased-whole-word-masking --do_train --do_eval --dataset_name squad --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --output_dir bert_squad_int8 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=10 --save_steps=400 --nncf_config nncf_bert_config_squad.json`

_INT8 model (symmetric quantization) + Knowledge Distillation_ - 92.89% F1, 86.68% EM on the dev set.

**INT8 model quantization-aware training + Knowledge Distillation command line (trained on 4x Tesla V100):**

`python examples/pytorch/question-answering/run_qa.py --model_name_or_path bert-large-uncased-whole-word-masking --do_train --do_eval --dataset_name squad --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --output_dir bert_squad_int8 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=10 --save_steps=400 --nncf_config nncf_bert_config_squad_kd.json`

**Fine-tuned INT8 model evaluation and ONNX export command line:**

`python examples/pytorch/question-answering/run_qa.py --model_name_or_path bert_squad_int8 --do_eval --dataset_name squad --max_seq_length 384 --doc_stride 128 --output_dir bert_squad_int8 --per_gpu_eval_batch_size=1 --nncf_config nncf_bert_config_squad.json --to_onnx bert_squad_int8.onnx`


### BERT-CoNLL2003

_Full-precision FP32 baseline model_ - bert-base-cased model, trained on CoNLL2003 - 99.17% acc, 95.03% F1

_INT8 model (symmetric quantization)_ - 99.18% acc, 95.31% F1

**INT8 model quantization-aware training command line (trained on 4x Tesla V100):**

`python examples/pytorch/token-classification/run_ner.py --model_name_or_path *path_to_fp32_finetuned_model* --dataset_name conll2003 --output_dir bert_base_cased_conll_int8 --do_train --do_eval --save_strategy epoch --evaluation_strategy epoch --nncf_config nncf_bert_config_conll.json`


**Fine-tuned INT8 model evaluation and ONNX export command line:**

`python examples/pytorch/token-classification/run_ner.py --model_name_or_path bert_base_cased_conll_int8 --dataset_name conll2003 --output_dir bert_base_cased_conll_int8 --do_eval --nncf_config nncf_bert_config_squad.json --to_onnx bert_base_cased_conll_int8.onnx`


### BERT-MRPC

_Full-precision FP32 baseline model_ -  bert-base-cased-finetuned-mrpc, 84.56% acc

_INT8 model (symmetric quantization)_ - 84.8% acc

**INT8 model quantization-aware training command line (trained on 1x RTX 2080):**

`python examples/pytorch/token-classification/run_glue.py --model_name_or_path bert-base-cased-finetuned-mrpc --task_name mrpc --do_train --do_eval --num_train_epochs 5.0 --per_device_eval_batch_size 1 --output_dir bert_cased_mrpc_int8 --evaluation_strategy epoch --save_strategy epoch --nncf_config nncf_bert_config_mrpc.json`

**Fine-tuned INT8 model evaluation and ONNX export command line:**

`python examples/pytorch/token-classification/run_ner.py --model_name_or_path bert_cased_mrpc_int8 --task_name mrpc --do_eval --per_gpu_eval_batch_size 1 --output_dir bert_cased_mrpc_int8 --nncf_config nncf_bert_config_mrpc.json --to_onnx bert_base_cased_mrpc_int8.onnx`

### RoBERTA-MNLI

_Full-precision FP32 baseline model_ - roberta-large-mnli, pre-trained on MNLI - 90.6% accuracy (matched), 90.1% accuracy (mismatched)

_INT8 model (asymmetrically quantized)_ - 89.25% accuracy (matched), 88.9% accuracy (mismatched)

**INT8 model quantization-aware training command line:**

`python examples/pytorch/text-classification/run_glue.py --model_name_or_path roberta-large-mnli --task_name mnli --do_train --do_eval --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 1 --learning_rate 2e-5 --num_train_epochs 3.0 --max_seq_length 128 --output_dir roberta_mnli_int8 --save_steps 400 --nncf_config nncf_roberta_config_mnli.json`


**Fine-tuned INT8 model evaluation and ONNX export command line:**

`python examples/pytorch/text-classification/run_glue.py --model_name_or_path roberta_mnli_int8 --task_name mnli --do_eval --learning_rate 2e-5 --num_train_epochs 3.0 --max_seq_length 128 --per_gpu_eval_batch_size 1 --output_dir roberta_mnli_int8 --save_steps 400 --nncf_config nncf_roberta_config_mnli.json --to_onnx roberta_mnli_int8.onnx`


### DistilBERT-SST-2

_Full-precision FP32 baseline model_ - distilbert-base-uncased-finetuned-sst-2-english, pre-trained on SST-2 - 91.1% accuracy

_INT8 model (symmetrically quantized)_ - 90.94% accuracy

**INT8 model quantization-aware training command line:**

`python examples/pytorch/text-classification/run_glue.py --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english --task_name sst2 --do_train --do_eval --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 1 --learning_rate 5e-5 --num_train_epochs 3.0 --max_seq_length 128 --output_dir distilbert_sst2_int8 --save_steps 100000 --nncf_config nncf_distilbert_config_sst2.json`


**Fine-tuned INT8 model evaluation and ONNX export command line:**

`python examples/pytorch/text-classification/run_glue.py --model_name_or_path distilbert_sst2_int8 --task_name sst2 --do_eval --per_gpu_eval_batch_size 1 --max_seq_length 128 --output_dir distilbert_sst2_int8 --save_steps 100000 --nncf_config nncf_distilbert_config_sst2.json --to_onnx distilbert_sst2_int8.onnx`


### MobileBERT-SQuAD v1.1

_Full-precision FP32 baseline model_ - google/mobilebert-uncased, trained on SQuAD v1.1 - 89.98% F1, 82.61% EM on the dev set,

_INT8 model (symmetric quantization)_ - 89.4% F1, 82.05% EM on the dev set.

**INT8 model quantization-aware training command line (trained on 3x Tesla V100):**

`python examples/pytorch/question-answering/run_qa.py --model_name_or_path <path_to_pretrained_mobilebert_squad> --do_train --do_eval --dataset_name squad --learning_rate 3e-5 --num_train_epochs 5 --max_seq_length 384 --doc_stride 128 --output_dir mobilebert_squad_int8 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=6 --save_steps=400 --nncf_config nncf_mobilebert_config_squad_int8.json`

**Fine-tuned INT8 model evaluation and ONNX export command line:**

`python examples/pytorch/question-answering/run_qa.py --model_name_or_path mobilebert_squad_int8 --do_eval --dataset_name squad --max_seq_length 384 --doc_stride 128 --output_dir mobilebert_squad_int8 --per_gpu_eval_batch_size=1 --nncf_config nncf_mobilebert_config_squad_int8.json --to_onnx mobilebert_squad_int8.onnx`

### GPT-2-WikiText 2 (raw) language modeling

_Full-precision FP32 baseline model_ - 19.73 perplexity on the test set

_INT8 model (symmetric quantization)_ - 20.9 perplexity on the test set


**INT8 model quantization-aware training command line (trained on 1x Tesla V100):**

`python examples/pytorch/language-modeling/run_clm.py --model_name_or_path <path_to_pretrained_gpt2_on_wikitext2> --do_train --do_eval --dataset_name wikitext --num_train_epochs 3 --output_dir gpt2_wikitext2_int8 --per_gpu_eval_batch_size=1 --per_gpu_train_batch_size=4 --save_steps=591 --nncf_config nncf_gpt2_config_wikitext_hw_config.json`

**Fine-tuned INT8 model evaluation and ONNX export command line:**

`python examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2_wikitext2_int8 --do_eval --dataset_name wikitext --output_dir gpt2_wikitext2_int8 --per_gpu_eval_batch_size=1 --nncf_config nncf_gpt2_config_wikitext_hw_config.json --to_onnx gpt2_wikitext2_int8.onnx`

