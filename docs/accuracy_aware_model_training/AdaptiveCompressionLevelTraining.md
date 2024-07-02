# Adaptive Compression Level training loop in NNCF

Adaptive Compression Level training loop is the meta-algorithm that performs searching for the most compression level of the underneath compression algorithms while staying within the range of the user-defined maximum accuracy degradation.
The compression pipeline can consist of several compression algorithms (Algorithms Mixing), however, **performing a compression level search is supported only for a single compression algorithm with an adaptive compression level**. They could be  **Magnitude Sparsity** and **Filter Pruning**. In the other words, the compression schemes like **Quantization** + **Filter Pruning** or **Quantization** + **Sparsity** are supported, while **Filter Pruning** + **Sparsity** is not, because **Filter Pruning** and **Sparsity** both are algorithms with adaptive compression level.

See a PyTorch [example](../../examples/torch/classification/main.py) for **Quantization** + **Filter Pruning** scenario on CIFAR10 and ResNet18 [config](../../examples/torch/classification/configs/pruning/resnet18_cifar10_accuracy_aware.json).

The exact compression algorithm for which the compression level search will be applied is determined in "compression" config section. The parameters to be set by the user in this config section are:

1. `maximal_relative_accuracy_degradation` or `maximal_absolute_accuracy_degradation` (Optional; default `maximal_relative_accuracy_degradation=1.0`) - the maximal allowed accuracy metric drop relative to the original model metrics (in percent) or the maximal allowed absolute accuracy metric drop (in original metrics value),
2. `initial_training_phase_epochs` (Optional; default=5) - number of epochs to train the model with the compression schedule specified in the `"params"` section of `"compression"` algorithm.
3. `patience_epochs` (Optional; default=3) - number of epochs to train the model for a compression rate level set by the search algorithm before switching to another compression rate value.
4. `minimal_compression_rate_step` (Optional; default=0.025) - the minimal compression rate change step value after which the training loop is terminated.
5. `initial_compression_rate_step` (Optional; default=0.1) - initial value for the compression rate increase/decrease training phase of the compression training loop.
6. `compression_rate_step_reduction_factor` (Optional; default=0.5) - factor used to reduce the compression rate change step in the adaptive compression training loop.
7. `lr_reduction_factor` (Optional; default=0.5) - factor used to reduce the base value of the learning rate scheduler after compression rate step is reduced.
8. `maximal_total_epochs` (Optional; default=10000) - number of training epochs, if the fine-tuning epoch reaches this number, the loop finishes the fine-tuning and return the model with thi highest compression rate and the least accuracy drop.

To launch the adaptive compression training loop, the user should define several functions related to model training, validation and optimizer creation (see [the usage documentation](../usage/training_time_compression/other_algorithms/Usage.md#accuracy-aware-model-training) for more details) and pass them to the run method of an `AdaptiveCompressionTrainingLoop` instance.
The training loop logic inside of the `AdaptiveCompressionTrainingLoop` is framework-agnostic, while all of the framework specifics are encapsulated inside of corresponding `Runner` objects, which are created and called inside the training loop.
The adaptive compression training loop is generally aimed at automatically searching for the optimal compression rate in the model, with the parameters of the search algorithm specified in the configuration file.
Below is an example of a filter pruning configuration with added `"accuracy_aware_training"` parameters.

```json5
{
    "input_infos": {"sample_size": [1, 2, 224, 224]},
    "accuracy_aware_training": {
        "mode": "adaptive_compression_level",
        "params": {
            "maximal_relative_accuracy_degradation": 1.0, // Optional
            "initial_training_phase_epochs": 100, // Optional
            "patience_epochs": 30, // Optional
            "minimal_compression_rate_step": 0.025, // Optional
            "initial_compression_rate_step": 0.1, // Optional
            "compression_rate_step_reduction_factor": 0.5, // Optional
            "lr_reduction_factor": 0.5, // Optional
            "maximal_total_epochs": 10000 // Optional
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

## How Adaptive Compression Level training loop works

The first step is **Initial Training Phase** - It corresponds to the amount of epochs that the model is going to be trained for with the initial compression rate level/schedule set by the user in the standard NNCF manner (the initial pruning rate schedule above is an exponential schedule with the target pruning rate of 0.1).

The second one is **Finding the optimal compression rate**, where the next compression rate value is determined by the search algorithm and the model is fine-tuned for a maximum of `"patience_epochs"` number of epochs. Fine-tuning may end earlier, if the accuracy criteria has been reached. The process is continued until the search algorithm terminates. The returned model is the model with the highest compression rate encountered, which satisfies the accuracy drop criterion - the accuracy drop of the compressed model should not be more than `"maximal_relative_accuracy_degradation`" or "`maximal_absolute_accuracy_degradation`".

## Compression rate search algorithm

The default behavior for the compression rate search algorithm implies changes in the compression rate level value by a step value that is decreasing throughout training.
The training is terminated once the compression rate step value reaches the minimal value determined by the `"minimal_compression_rate_step"` parameter that can be specified in the `"params"` of `"accuracy_aware_training"` section.
The initial value for the compression rate step is given by the `"initial_compression_rate_step"` parameter.
The step value is decreased by the `"compression_rate_step_reduction_factor"` value at points throughout training whenever the direction of change in compression rate changes at a point where the new compression rate is selected.
That is, if a too big of an increase in compression rate resulted in the accuracy metrics below the user-defined criterion, the compression rate is reduced by a lower step in an attempt to restore the accuracy and vice versa, if the decrease was sufficient to satisfy the accuracy criterion, the compression rate is increased by a lower step to check if this higher compression rate could also result in tolerable accuracy values.
This sequential search is limited by the minimal granularity of the steps given by `"minimal_compression_rate_step"`.

## Example

An example of how model is compressed using Adaptive Compression Training Loop is given on the figure below.
![Example](actl_progress_plot.png)
