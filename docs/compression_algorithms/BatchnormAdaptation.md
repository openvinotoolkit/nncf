#### Batch-norm statistics adaptation

After the compression-related changes in the model have been committed, the statistics of the batchnorm layers (per-channel rolling means and variances of activation tensors) can be updated by passing several batches of data through the model before the fine-tuning starts. 
This allows to correct the compression-induced bias in the model and reduce the corresponding accuracy drop even before model training. 
This option is common for quantization, magnitude sparsity and filter pruning algorithms. 
It can be enabled by setting a non-zero value of `num_bn_adaptation_samples` in the `batchnorm_adaptation` section of the `initializer` configuration - see [NNCF config schema](FIXME) for reference.

Note that in order to use batchnorm adaptation for your model, you must supply to NNCF a data loader using a `register_default_init_args` helper function or by registering a `nncf.config.structures.BNAdaptationInitArgs` structure within the `NNCFConfig` object in your integration code.

