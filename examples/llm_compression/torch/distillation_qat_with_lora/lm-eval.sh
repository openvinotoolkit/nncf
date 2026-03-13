IR_DIR=u2_u4_ov_model
lm_eval \
--model openvino \
--model_args pretrained=$IR_DIR \
--device cpu \
--output_path ov_eval \
--limit 100 \
--tasks lambada_openai
