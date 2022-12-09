# Integrating BootstrapNAS into Transformers
https://github.com/huggingface/transformers

This folder contains a git patch to enable NNCF-based neural architecture search for XNLI, SQuAD and GLUE training pipelines of the huggingface transformers repository. 

Instructions:
1. Apply the `0001-Modifications-for-bootstrapNAS-usage.patch` file to the huggingface transformers repository checked out at commit id: `bd469c40659ce76c81f69c7726759d249b4aef49`

2. Install the `transformers` library and the example scripts from the patched repository as described in the documentation for the huggingface transformers repository.

3. To start neural architecture search of NLP models using BootstrapNAS, use the regular scripts and command line parameters for training, but with additional `--nncf_config <path_to_nncf_config>` parameter.
The NNCF configs to be used in this way are also provided in the same patch on a per-model basis.


## Training Script (Trained on 1x Tesla V100)


### BERT-XNLI

    python examples/pytorch/text-classification/run_xnli.py \
        --model_name_or_path bert-base-chinese \
        --language zh \
        --train_language zh \
        --do_train \
        --do_eval \
        --do_search \
        --per_gpu_train_batch_size 48 \
        --per_gpu_eval_batch_size 64 \
        --max_seq_length 128 \
        --output_dir results/bert_xnli_nas \
        --nncf_config nncf_bootstrapnas_config/nncf_bootstrapnas_bert_config_xnli.json \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --metric_for_best_model accuracy

### BERT-SQuAD v1.1


    python examples/pytorch/question-answering/run_qa.py \
        --model_name_or_path bert-large-uncased-whole-word-masking \
        --do_train \
        --do_eval \
        --do_search \
        --dataset_name squad \
        --per_gpu_train_batch_size 12 \
        --per_gpu_eval_batch_size 64 \
        --output_dir results/bert_squad_nas \
        --max_seq_length 384 \
        --doc_stride 128 \
        --nncf_config nncf_bootstrapnas_config/nncf_bootstrapnas_bert_config_squad.json \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --metric_for_best_model f1

### BERT-CoNLL2003

    python examples/pytorch/token-classification/run_ner.py \
        --model_name_or_path <path-to-pretrained-model> \
        --dataset_name conll2003 \
        --do_train \
        --do_eval \
        --do_search \
        --output_dir results/bert_conll_nas \
        --nncf_config nncf_bootstrapnas_config/nncf_bootstrapnas_bert_config_conll.json \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --metric_for_best_model accuracy


### BERT-MRPC

    python examples/pytorch/text-classification/run_glue.py \
        --model_name_or_path bert-base-cased-finetuned-mrpc \
        --task_name mrpc \
        --do_train \
        --do_eval \
        --do_search \
        --output_dir results/bert_mrpc_nas \
        --nncf_config nncf_bootstrapnas_config/nncf_bootstrapnas_bert_config_mrpc.json \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --metric_for_best_model accuracy
