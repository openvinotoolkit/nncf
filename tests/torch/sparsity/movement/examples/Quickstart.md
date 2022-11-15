# Movement example runs

This folder presents some examples for movement sparsity.

## GLUE-MRPC task

```
TASK_NAME=mrpc
NNCF_CONFIG=./bert_tiny_uncased_mrpc_movement.json
python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --logging_steps 50 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --output_dir /tmp/$TASK_NAME/ \
  --nncf_config $NNCF_CONFIG
```

