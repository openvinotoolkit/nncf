# Early Exit training loop in NNCF

Early Exit training loop aims to get the compressed model with the desired accuracy criteria as earlier as possible. This is done by checking a compressed model accuracy after each training epoch step and also after the initialization step then exits the fine-tuning process once the accuracy reaches the user-defined criteria
This pipeline is simple but effective. It reduces a fine-tuning time for many models till just an initialization step.

Note: since the EarlyExit training does not control any compression parameter the specified accuracy criterium cannot be satisfied in some cases

This training loop supports any combination of NNCF compression algorithms.

There are only two main parameters of Early-Exit training loop: `maximal_relative_accuracy_degradation` or `maximal_absolute_accuracy_degradation` - relative/absolute accuracy drop in percentage/in original metric with original, uncompressed model less than that is user tolerant. And `maximal_total_epochs` - number of training epochs, if the fine-tuning epoch reaches this number, the loop finishes the fine-tuning and returns the model with the least accuracy drop.

Example of config file needed to be provided to create_accuracy_aware_training_loop (see [the usage documentation](../usage/training_time_compression/other_algorithms/Usage.md#accuracy-aware-model-training) for more details).

```json5
{
    "input_infos": {"sample_size": [1, 2, 224, 224]},
    "accuracy_aware_training": {
        "mode": "early_exit",
        "params": {
            "maximal_relative_accuracy_degradation": 1.0,
            "maximal_total_epochs": 100
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
