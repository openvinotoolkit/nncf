# Movement Sparsity

[Movement Pruning (Sanh et al., 2020)](https://arxiv.org/pdf/2005.07683.pdf) is an effective learning-based unstructured sparsification algorithm, especially for Transformer-based models in transfer learning setup. [Lagunas et al., 2021](https://arxiv.org/pdf/2109.04838.pdf) extends the algorithm to sparsify by block grain size, enabling structured sparsity which can achieve device-agnostic inference acceleration.

NNCF implements both unstructured and structured movement sparsification. The implementation is designed with a minimal set of configuration for ease of use. The algorithm can be applied in conjunction with other NNCF algorithms, e.g. quantization-aware training and knowledge distillation. The optimized model can be deployed and accelerated via [OpenVINO](https://docs.openvino.ai/2024/index.html) toolchain.

For usage explanation of the algorithm, let's start with an example configuration below which is targeted for BERT models.

## Example configuration of Movement Sparsity for BERT models

```json
{
    "compression":
    {
    "algorithm":  "movement_sparsity",
    "params": {
        "warmup_start_epoch":  1,
        "warmup_end_epoch":    4,
        "importance_regularization_factor":  0.01,
        "enable_structured_masking":  true
    },
    "sparse_structure_by_scopes": [
        {"mode":  "block",   "sparse_factors": [32, 32], "target_scopes": "{re}.*BertAttention.*"},
        {"mode":  "per_dim", "axis":  0,                 "target_scopes": "{re}.*BertIntermediate.*"},
        {"mode":  "per_dim", "axis":  1,                 "target_scopes": "{re}.*BertOutput.*"},
    ],
    "ignored_scopes": ["{re}.*NNCFEmbedding", "{re}.*qa_outputs*", "{re}.*LayerNorm.*"]
    }
}
```

<p align="center">
    <img src="movement_sparsity_lifecycle.jpg" alt="movement sparsity lifecycle"/>
</p>

This diagram is the sparsity level of BERT-base model over the optimization lifecycle with the configuration above. In essence, the algorithm consists of two major stages:

1. **Unstructured sparsification**: In the first stage, model weights are gradually sparsified in the grain size specified by `sparse_structure_by_scopes`. This example will result in _BertAttention layers (Multi-Head Self-Attention)_ being sparsified in 32 by 32 block size, whereas _BertIntermediate, BertOutput layers (Feed-Forward Network)_ will be sparsified in its row or column respectively. The sparsification follows a predefined warmup schedule where users only have to specify the start `warmup_start_epoch` and end `warmup_end_epoch` and the sparsification strength proportional to `importance_regularization_factor`. Users might need some heuristics to find a satisfactory trade-off between sparsity and task performance. For more details on how movement sparsification works, please refer the original papers [1, 2] .

2. **Structured masking and fine-tuning**: At the end of first stage, i.e. `warmup_end_epoch`, the sparsified model cannot be accelerated without tailored HW/SW but some sparse structures can be totally discarded from the model to save compute and memory footprint. NNCF provides mechanism to achieve structured masking by `"enable_structured_masking": true`, where it automatically resolves the structured masking between dependent layers and rewinds the sparsified parameters that does not participate in acceleration for task modeling. In the example above, the sparsity level has dropped after `warmup_end_epoch` due to structured masking and the model will continue to fine-tune thereafter. Currently, the automatic structured masking feature was tested on **_BERT, DistilBERT, RoBERTa, MobileBERT, Wav2Vec2, Swin, ViT, CLIPVisual_** architectures defined by [Hugging Face&#39;s transformers](https://huggingface.co/docs/transformers/index). Support for other architectures is not guaranteed. Users can disable this feature by setting `"enable_structured_masking": false`, where the sparse structures at the end of first stage will be frozen and training/fine-tuning will continue on unmasked parameters. Please refer next section to realize model inference acceleration with [OpenVINO](https://docs.openvino.ai/2024/index.html) toolchain.

## Inference Acceleration via [OpenVINO](https://docs.openvino.ai/2024/index.html)

Optimized models are compatible with OpenVINO toolchain. Use `compression_controller.export_model("movement_sparsified_model.onnx")` to export model in onnx format. Sparsified parameters in the onnx are in value of zero. Structured sparse structures can be discarded during ONNX translation to OpenVINO IR using [Model Conversion](https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-to-ir.html) with utilizing [pruning transformation](https://docs.openvino.ai/2024/documentation/legacy-features/transition-legacy-conversion-api.html#transform). Corresponding IR is compressed and deployable with [OpenVINO Runtime](https://docs.openvino.ai/2024/openvino-workflow/running-inference.html). To quantify inference performance improvement, both ONNX and IR can be profiled using [Benchmark Tool](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html).

## Getting Started

Please refer [optimum-intel](https://github.com/huggingface/optimum-intel/tree/main/examples/openvino) for example pipelines on image classification, question answering, etc. The repository also provides examples of joint pruning, quantization and distillation, end-to-end from NNCF optimization to compressed OpenVINO IR.

## Known Limitation

1. Movement sparsification only supports `torch.nn.Linear` layers.
2. Automatic structured masking feature supports **BERT, DistilBERT, RoBERTa, MobileBERT, Wav2Vec2, Swin, ViT, CLIPVisual** architectures defined by [Hugging Face&#39;s transformers](https://huggingface.co/docs/transformers/index). Other similar architectures may work, but support is not guaranteed.

## Detailed description of Movement Sparsity configuration

- `algorithm`: The algorithm name is "movement_sparsity".
- `warmup_start_epoch` & `warmup_end_epoch`: The algorithm will conduct model weight sparsification gradually from epoch >= `warmup_start_epoch` to epoch < `warmup_end_epoch`, with epoch is zero-indexed. This span is known as sparsification warm-up (stage 1).
- `importance_regularization_factor`: The regularization factor on weight importance scores. With a larger positive value, more model weights will be regarded as less important and thus be sparsified. The appropriate value range of this argument can be quite different per model and task.
- `enable_structured_masking`: Optional. A boolean to enable structured mask resolution after warm-up stage. Setting it to `false` results in unstructured sparse output model. Default is `true`.
- `sparse_structure_by_scopes`: Describes how a layer will be sparsified. The value of the argument is a list, where each entry dictionary must specify `mode` and `target_scopes`, together with additional key-pair for a given mode (e.g., `sparse_factors` for "block" mode, and `axis` for "per_dim" mode).

  - Supported options for `mode`:
    - `fine`: Each weight element is learned individually whether to be sparsified. No extra argument needed.
    - `block`: Each block within a weight will be preserved or sparsified together as a whole. Requires `sparse_factors` to decide the block shape. Note that the block shape must be dividable w.r.t. the weight shape.
    - `per_dim`: The weight will be sparsified by a certain dimension. Requires `axis` to decide which dimension to sparsify. For example, for a linear layer containing a 2D weight matrix, `axis=0` means to be sparse by row, and `axis=1` means sparse by column.
  - `target_scopes`: A string or a list of strings representing the layers to sparsify with the specified `mode`. Value can be a complete scope name in NNCF Graph, or a regular expression specification starting with `"{re}"`. A supported layer is by default applied with `fine` mode if it is not specified in `sparse_structure_by_scopes`.

- `ignored_scopes`: A string or a list of strings representing the layers to be ignored by Movement Sparsity algorithm.

## Extra configuration in `params` section

Following arguments have been defaulted to work well out of the box. However, you can specify them for a more controlled sparsification strategy.

- `initial_importance_threshold` & `final_importance_threshold`: Optional. In Movement Sparsity, a weight will be masked if an importance score is lower than the threshold. The threshold gradually increases from `initial_importance_threshold` to `final_importance_threshold` during warm-up stage. By default, `final_importance_threshold` is set to 0, and `initial_importance_threshold` is adaptively determined at the beginning of warm-up such that the model is at about 0.1% sparsity level in sparsifying layers. Specifying these arguments customizes starting and ending importance threshold values.
- `power`: Optional. The importance threshold and regularization factor follow a concave polynomial warm-up schedule where its decay factor is parameterized by `power`. Default is 3.
- `steps_per_epoch`: Optional. Number of steps per epoch is needed for threshold and regularization factor scheduling. It varies by dataset size and training hyperparameters. By default, this can be automatically derived during the first epoch without any side effect, as long as `warmup_start_epoch` >= 1. Specification of `steps_per_epoch` is only required when warm-up sparsification is intended to start at the first epoch.

## References

1. Victor Sanh, Thomas Wolf, and Alexander M. Rush. 2020. [Movement Pruning: Adaptive Sparsity by Fine-Tuning](https://arxiv.org/pdf/2005.07683.pdf). In Advances in Neural Information Processing Systems, 33, pp. 20378-20389.
2. François Lagunas, Ella Charlaix, Victor Sanh, and Alexander M. Rush. 2021. [Block Pruning For Faster Transformers](https://arxiv.org/pdf/2109.04838.pdf). In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 10619–10629.
