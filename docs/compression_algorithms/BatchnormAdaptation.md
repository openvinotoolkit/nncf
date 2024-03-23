# Batch-norm statistics adaptation

After the compression-related changes in the model have been committed, the statistics of the batchnorm layers (per-channel rolling means and variances of activation tensors) can be updated by passing several batches of data through the model before the fine-tuning starts.
This allows to correct the compression-induced bias in the model and reduce the corresponding accuracy drop even before model training.
This option is common for quantization, magnitude sparsity and filter pruning algorithms.
It can be enabled by setting a non-zero value of `num_bn_adaptation_samples` in the `batchnorm_adaptation` section of the `initializer` configuration - see [NNCF config schema](https://openvinotoolkit.github.io/nncf/) for reference.

Note that in order to use batchnorm adaptation for your model, you must supply to NNCF a data loader using a `register_default_init_args` helper function or by registering a `nncf.config.structures.BNAdaptationInitArgs` structure within the `NNCFConfig` object in your integration code.

## Example configuration files

>_For the full list of the algorithm configuration parameters via config file, see the corresponding section in the [NNCF config schema](https://openvinotoolkit.github.io/nncf/)_.

- Apply batchnorm adaptation for 2048 samples (rounded to nearest batch size multiple) during model quantization:

```json5
{
    "input_info": {"sample_size" :  [1, 3, 224, 224]},  // the input shape of your model may vary
   "compression": {
        "algorithm": "quantization",
        "initializer": {
            "batchnorm_adaptation": {
                "num_bn_adaptation_samples": 2048
            }
        }
    }
}
```

- Apply batchnorm adaptation for 32 samples (rounded to nearest batch size multiple) during model magnitude-based sparsification:

```json5
{
    "input_info": {"sample_size" :  [1, 3, 224, 224]},  // the input shape of your model may vary
   "compression": {
        "algorithm": "magnitude_sparsity",
        "initializer": {
            "batchnorm_adaptation": {
                "num_bn_adaptation_samples": 32
            }
        },
       "params": {
           "sparsity_target": 0.5,
           "sparsity_target_epoch": 10
       }
    }
}
```
