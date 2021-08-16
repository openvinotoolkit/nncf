# Adaptive Compression Level training loop in NNCF

To launch the adaptive compression training loop, the user is expected to define several function related to model training, validation and optimizer creation (see [the usage documentation](../Usage.md#accuracy-aware-model-training) for more details) and pass them to the run method of an `AdaptiveCompressionTrainingLoop` instance. The training loop logic inside of the `AdaptiveCompressionTrainingLoop` is framework-agnostic, while all of the framework specifics are encapsulated inside of corresponding `Runner` objects, which are created and called inside the training loop. The adaptive compression training loop is generally aimed at automatically searching for the optimal compression rate in the model, with the parameters of the search algorithm specified in the configuration file as follows:
```
{
    "accuracy_aware_training": {
        "mode": "adaptive_compression_level",
        "params": {
            "maximal_accuracy_degradation": 1.0,
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
The above compression configuration implies that the compression rate to be varied during training is the filter pruning ratio, since the `"accuracy_aware_training"` section is specified inside the filter pruning algorithm configuration. The `"initial_training_phase_epochs"` parameter corresponds to the amount of epochs that the model is going to be trained for with the initial compression rate level/schedule set by the user in the standard NNCF manner (the initial pruning rate schedule above is an exponential schedule with the target pruning rate of 0.1). After this initial phase of fine-tuning, the next compression rate value is determined by the search algorithm and the model is fine-tuned with that selected compression rate value for `"patience_epochs"` number of epochs. The process is continued until after the search algorithm terminates. The returned model is the model with the highest compression rate encountered during training given that is satisfies the accuracy drop criterion -- the accuracy value of the compressed model should not be more that `"maximal_accuracy_degradation`" percent less that the original uncompressed model's accuracy value.
The default behavior for the compression rate search algorithm implies changes in the compression rate level value by a step value that is decreasing throughout training. The training is terminated once the compression rate step value reaches the minimal value determined by the `"minimal_compression_rate_step"` parameter that can be specified in the `"accuracy_aware_training"` section of the config (default value is 0.025). The initial value for the compression rate step is given be the `"compression_rate_step"` parameter and is equal to 0.1 by default. The step value is decreased by the `"step_reduction_factor"` value (0.5 by default) at points throughout training whenever the direction of change in compression rate changes at a point where the new compression rate is selected. That is, if a too big of an increase in compression rate resulted in the accuracy metrics below the user-defined criterion, the compression rate is reduced by a lower step in an attempt to restore the accuracy and vice versa, if the decrease was sufficient to satisfy the accuracy criterion, the compression rate is increased by a lower step to check if this higher compression rate could also result in tolerable accuracy values. This sequential search is limited by the minimal granularity of the steps given by `"minimal_compression_rate_step"`.
