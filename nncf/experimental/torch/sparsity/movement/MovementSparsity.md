### Movement Sparsity

Movement sparsity (Sanh et al., 2020) has proven to be an effective learning-based unstructured sparsity algorithm, especially for Transformer models in transfer learning. Its extension of block pruning (Lagunas et al., 2021) leverages the idea for structured pruning to accelerate Transformer inference.

Implementation of Movement Sparsity in NNCF collapses unstructured sparsification and structured pruning into a single loop.

```
TODO(VS): illustration
```

The algorithm consists of (1) an unstructured sparsity warmup stage, when model weights are gradually sparsified by customized grid sizes; (2) an optional structured mask resolving process intended to accelerate Transformer blocks; (3) a fine-tuning stage with fixed weight sparsification.

#### Configuration

The json configuration to enable Movement Sparsity is as follows. It is also possible to apply Movement Sparsity together with quantization and/or knowledge distillation.

```json
{
  "compression": [
    {
      "algorithm": "movement_sparsity",
      "params": {
        "warmup_start_epoch": 1,
        "warmup_end_epoch": 6,
        "importance_regularization_factor": 0.1,
        "enable_structured_masking": false
      },
      "sparse_structure_by_scopes": [
        {
          "mode": "fine",
          "target_scopes": "BertForSequenceClassification/NNCFLinear[classifier]/linear_0"
        },
        {
          "mode": "block",
          "sparse_factors": [64, 64],
          "target_scopes": ["{re}.*attention.*", "{re}.*another_block.*"]
        },
        {
          "mode": "per_dim",
          "axis": 0,
          "target_scopes": "{re}.*intermediate.*"
        }
      ],
      "ignored_scopes": ["{re}.*pooler.*", "{re}.*classifier.*"]
    },
    ...(other algorithms)
  ]
}
```

**Basic configuration**

Note: currently only `torch.nn.Linear` layers are supported by Movement Sparsity.

- `algorithm`: The algorithm name should be "movement_sparsity".

- `warmup_start_epoch` & `warmup_end_epoch`: The algorithm will conduct model weight sparsification gradually from epoch_index=`warmup_start_epoch` (include) to epoch_index=`warmup_end_epoch` (exclude), with epoch_index starting from 0. This stage here is called "warmup". In the example config above, the warmup stage starts at 2nd epoch and lasts for 5 epochs.

- `importance_regularization_factor`: The regularization factor on weight importance scores. With a larger positive value, more model weights will be regarded as less important and thus be sparsified. The appropriate value range of this argument can be quite different per model and task.

- `enable_structured_masking`: Optional. A boolean to decide whether to do structured mask resolution after warmup stage. Currently, we only support structured masking on multi-head self-attention blocks and feed-forward networks in BERT, Swin and Wav2vec2 implemented in [Transformers](https://github.com/huggingface/transformers). If it is set to `false`, the output model is instead with unstructured sparsity. Default is `false`.

- `sparse_structure_by_scopes`: Describes how each supported layer will be sparsified. This argument is a list, where each entry dictionary must specify `mode` and `target_scopes`, together with some extra arguments for a certain mode (e.g., `sparse_factors` for "block", and `axis` for "per_dim").

  - Supported options for `mode`:

    - fine: Each weight element is learned individually whether to be sparsified. No extra argument needed.

    - block: Each block within a weight will be preserved or sparsified together as a whole. Requires `sparse_factors` to decide the block shape. Note that the block shape must be dividable w.r.t. the weight shape.

    - per_dim: The weight will be sparsified by a certain dimension. Requires `axis` to decide which dimension to sparsify. Typically, for a layer containing a 2D weight matrix, `axis=0` means to be sparse by row, and `axis=1` means sparse by column.

  - `target_scopes`: A string or a list of strings representing the layers to sparsify with the specified `mode`. Can be a complete scope name in NNCF Graph, or a regular expression starting with "{re}".

  If a supported layer is not mentioned in `sparse_structure_by_scopes`, by default we apply "fine" mode on it.

- `ignored_scopes`: A string or a list of strings representing the layers to ignore by Movement Sparsity. Note that layers other than `torch.nn.Linear` are now not supported (e.g., convolutions) and are thus not necessary to be mentioned here.

**Extra configuration in `params` section**

Usually we do not need to manually set the following arguments, but you can specify them for a more flexible sparsification strategy.

- `initial_importance_threshold` & `final_importance_threshold`: Optional. In Movement Sparsity, a weight will be sparsified if the corresponding importance score is lower than threshold. The threshold gradually increases from `initial_importance_threshold` to `final_importance_threshold` during warmup stage. By default, `final_importance_threshold` is set to 0, and `initial_importance_threshold` is adaptively decided during training (usually a negative number) so that the model is with about 0.1% relative sparsity on involved layers at the beginning of warmup stage. User can let the threshold start from or end at a customized value.

- `power`: Optional. The threshold updates during warmup follow the concave polynomial decay with a certain `power`. Default is 3.

- `steps_per_epoch`: Optional. The threshold is updated each step during warmup, and thus `steps_per_epoch` is needed for calculation. By default, this can be automatically counted during 1st epoch without any side effect, as long as `warmup_start_epoch` >= 1. Setting `steps_per_epoch` is only required when you want to start sparsification from epoch 0.

For more information, please refer to the following publication:

```
TODO(VS) is there a public link?
```

#### References

- Victor Sanh, Thomas Wolf, and Alexander M. Rush. 2020. Movement Pruning: Adaptive Sparsity by Fine-Tuning. In Advances in Neural Information Processing Systems, 33, pp. 20378-20389.

- François Lagunas, Ella Charlaix, Victor Sanh, and Alexander M. Rush. 2021. Block Pruning For Faster Transformers. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 10619–10629.
