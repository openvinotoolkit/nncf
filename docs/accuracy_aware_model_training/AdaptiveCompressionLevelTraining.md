# Adaptive Compression Level training loop in NNCF

The compression level search can be only done for a single compression algorithm in the pipeline (i.e. several compression algorithms could be applied to the model and the search is going to be performed for a single one); currently supported algorithms for compression rate search are magnitude sparsity and filter pruning. 
The exact compression algorithm for which the search is done is determined from `"compression"` config section. Below is an example of a filter pruning configuration with added `"accuracy_aware_training"` parameters. The parameters to be set by the user in this config section are: 
1) `maximal_relative_accuracy_degradation` or `maximal_absolute_accuracy_degradation` - the maximal allowed accuracy metric drop relative to the original model metrics (in percent) or the maximal allowed absolute accuracy metric drop (in original metrics value),
2) `initial_training_phase_epochs` - the number of epochs to train the model with the compression schedule specified in the `"params"` section of `"compression"` algorithm. 

3) `patience_epochs` - the number of epochs to train the model for a compression rate level set by the search algorithm before switching to another compression rate value.
4) (Optional; default=0.025) `minimal_compression_rate_step` - The minimal compression rate change step value after which the training loop is terminated.
5) (Optional; default=0.1) `initial_compression_rate_step` - Initial value for the compression rate increase/decrease training phase of the compression training loop.
6) (Optional; default=0.5) `compression_rate_step_reduction_factor` - Factor used to reduce the compression rate change step in the adaptive compression training loop. 
4) (Optional; default=1) `validate_every_n_epochs` - The parameter specifies across which number of epochs `Runner` should validate the compressed model.
5) (Optional; default=1e4) `maximal_total_epochs` -  - number of training epochs, if the fine-tuning epoch reaches this number, the loop finishes the fine-tuning and return the model with the least accuracy drop.


To launch the adaptive compression training loop, the user is expected to define several function related to model training, validation and optimizer creation (see [the usage documentation](../Usage.md#accuracy-aware-model-training) for more details) and pass them to the run method of an `AdaptiveCompressionTrainingLoop` instance. The training loop logic inside of the `AdaptiveCompressionTrainingLoop` is framework-agnostic, while all of the framework specifics are encapsulated inside of corresponding `Runner` objects, which are created and called inside the training loop. The adaptive compression training loop is generally aimed at automatically searching for the optimal compression rate in the model, with the parameters of the search algorithm specified in the configuration file as follows:
```
{
    "accuracy_aware_training": {
        "mode": "adaptive_compression_level",
        "params": {
            "maximal_relative_accuracy_degradation": 1.0,
            "initial_training_phase_epochs": 100,
            "patience_epochs": 30
        }
    },
    "compression": [
        {
            "algorithm": "filter_pruning",
            "pruning_init": 0.05,
            "params": {
                "schedule": "exponential",
                "pruning_target": 0.1,
                "pruning_steps": 50,
                "weight_importance": "geometric_median"
            }
        }
    ]
}

```

## Description of the work of Adaptive Compression Level training loop

The first step is **Initial Training Phase** - It corresponds to the amount of epochs that the model is going to be trained for with the initial compression rate level/schedule set by the user in the standard NNCF manner (the initial pruning rate schedule above is an exponential schedule with the target pruning rate of 0.1).

The second one is **Finding the optimal compression rate**, where the next compression rate value is determined by the search algorithm and the model is fine-tuned with that selected compression rate value for `"patience_epochs"` number of epochs. The process is continued until after the search algorithm terminates. The returned model is the model with the highest compression rate encountered during training given that is satisfies the accuracy drop criterion -- the accuracy value of the compressed model should not be more that `"maximal_relative_accuracy_degradation`" percent less that the original uncompressed model's accuracy value.

## Compression rate search algorithm

The default behavior for the compression rate search algorithm implies changes in the compression rate level value by a step value that is decreasing throughout training.
The training is terminated once the compression rate step value reaches the minimal value determined by the `"minimal_compression_rate_step"` parameter that can be specified in the `"params"` of `"accuracy_aware_training"` section.
The initial value for the compression rate step is given be the `"initial_compression_rate_step"` parameter.
The step value is decreased by the `"compression_rate_step_reduction_factor"` value at points throughout training whenever the direction of change in compression rate changes at a point where the new compression rate is selected.
That is, if a too big of an increase in compression rate resulted in the accuracy metrics below the user-defined criterion, the compression rate is reduced by a lower step in an attempt to restore the accuracy and vice versa, if the decrease was sufficient to satisfy the accuracy criterion, the compression rate is increased by a lower step to check if this higher compression rate could also result in tolerable accuracy values.
This sequential search is limited by the minimal granularity of the steps given by `"minimal_compression_rate_step"`.

