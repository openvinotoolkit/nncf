# EarlyExit training loop in NNCF

Early-Exit training loop aims to get the compressed model with a desired accuracy criteria as earlier as possible. After NNCF initialization algorithms applied and then after each trainig epoch the loop measures accuracy of the compressed model and if it is reach the user-defined criteria exits and returns a compressed model.
This pipeline is simple but effective. For many models it reduces training time till just an initialization step. 

This training loop supports any combination of NNCF compression algorithms.

There is an example of config file needed to be provided to create_accuracy_aware_training_loop ((see [the usage documentation](../Usage.md#accuracy-aware-model-training) for more details)).

```
{
    "accuracy_aware_training": {
        "mode": "early_exit",
        "params": {
            "maximal_accuracy_degradation": 1.0,
            "maximal_total_expochs": 100,
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
 
There are only two parameters of Early-Exit training loop: maximal_relative_accuracy_degradation or maximal_absolute_accuracy_degradation- relative/absolute accuracy drop in percentage/in original metric with original, uncompressed model less than that is user tolerant. And maximal_total_epochs - number of training epochs, if the fine-tuning epoch reaches this number, the loop finishes the fine-tuning and return the model with the least accuracy drop 