### Pruning

#### Filter pruning

Filter pruning algorithm zeros output filters in Convolutional layers based on some filter importance criterion  (filters with smaller importance are pruned).
The framework contains three filter importance criteria: `L1`, `L2` norm, and `Geometric Median`. Also, different schemes of pruning application are presented by different schedulers.
Not all Convolution layers in the model can be pruned. Such layers are determined by the model architecture automatically as well as cross-layer dependencies that impose constraints on pruning filters.

#### Filter importance criteria **L1, L2**

 `L1`, `L2` filter importance criteria are based on the following assumption:
> Convolutional filters with small $l_p$ norms do not significantly contribute to output activation values, and thus have a small impact on the final predictions of CNN models.
In the above, the $l_p$ norm for filter F is:

$||F||_p = \sqrt[p]{\sum\limits\_{c, k_1, k_2 = 1}^{C, K, K} |F(c, k_1, k_2)|^p}$

During the pruning procedure filters with smaller  `L1` or `L2` norm will be pruned first.

**Geometric Median**

Usage of the geometric median filter importance criterion is based on the following assumptions:
> Let $\{F_i, \dots , F_j\}$ be the set of N filters in a convolutional layer that are closest to the geometric median of all the filters in that layer.   As it was shown, each of those filters can be decomposed into a linear combination of the rest of the filters further from the geometric median with a small error. Hence, these filters can be pruned without much impact on network accuracy.  Since we have only fixed number of filters in each layer and the task of calculation of geometric median is a non-trivial problem in computational geometry, we can instead find which filters minimize the summation of the distance with other filters.

Then Geometric Median importance of $F_i$ filter from $L_j$ layer is:
$G(F_i) = \sum_{F_j \in \{F_1, \dots F_m\}, j\neq i} ||F_i - F_j||_2$
Where $L_j$ is j-th convolutional layer in model. $\{F_1, \dots F_m\} \in L_j$ - set of all  output filters in $L_j$ layer.

Then during pruning filters with smaller $G(F_i)$ importance function will be pruned first.

#### Schedulers

**Baseline Scheduler**
 Firstly, during `num_init_steps` epochs the model is trained without pruning. Secondly, the pruning algorithm calculates filter importances and prunes a `pruning_target` part of the filters with the smallest importance in each prunable convolution.
The zeroed filters are frozen afterwards and the remaining model parameters are fine-tuned.

**Parameters of the scheduler:**
- `num_init_steps` - number of epochs for model pretraining **before** pruning.
- `pruning_target` - pruning level target. For example, the value `0.5` means that right after pretraining, convolutions that can be pruned will have 50% of their filters set to zero.


**Exponential scheduler**

Similar to the Baseline scheduler, during `num_init_steps` epochs model is pretrained without pruning.
During the next `pruning steps` epochs `Exponential scheduler` gradually increasing pruning level from `pruning_init` to `pruning_target`. After each pruning training epoch pruning algorithm calculates filter importances for all convolutional filters and prune (setting to zero) `current_pruning_rate` part of filters with the smallest importance in each Convolution.  After `num_init_steps` + `pruning_steps` epochs algorithm with zeroed filters is frozen and remaining model parameters only fine-tunes.

Current pruning level $P_{i}$ (on i-th epoch) during training calculates by equation:
$P_i = a * e^{- k * i}$
Where $a, k$ - parameters.

**Parameters of scheduler:**
- `num_init_steps` - number of epochs for model pretraining before pruning.
- `pruning_steps` - the number of epochs during which the pruning level target is increased from `pruning_init` to `pruning_target` value.
- `pruning_init` - initial pruning level target. For example, value `0.1` means that at the begging of training, convolutions that can be pruned will have 10% of their filters set to zero.
- `pruning_target` - pruning level target at the end of the schedule. For example, the value `0.5` means that at the epoch with the number of `num_init_steps + pruning_steps`, convolutions that can be pruned will have 50% of their filters set to zero.

**Exponential with bias scheduler**
Similar to the `Exponential scheduler`, but current pruning level $P_{i}$ (on i-th epoch) during training calculates by equation:
$P_i = a * e^{- k * i} + b$
Where $a, k, b$ - parameters.

> **NOTE**:  Baseline scheduler prunes filters only ONCE and after it just fine-tunes remaining parameters while exponential (and exponential with bias) schedulers choose and prune different filters subsets at each pruning epoch.

#### Batch-norm statistics adaptation

After the compression-related changes in the model have been committed, the statistics of the batchnorm layers
(per-channel rolling means and variances of activation tensors) can be updated by passing several batches of data
through the model before the fine-tuning starts. This allows to correct the compression-induced bias in the model
and reduce the corresponding accuracy drop even before model training. This option is common for quantization, magnitude
sparsity and filter pruning algorithms. It can be enabled by setting a non-zero value of `num_bn_adaptation_samples` in the
`batchnorm_adaptation` section of the `initializer` configuration (see example below).

#### Interlayer ranking types

Interlayer ranking type can be one of `unweighted_ranking` or `learned_ranking`.
- In case of `unweighted_ranking` and with  `all_weights=True` all filter norms will be collected together and sorted to choose the least important ones. But this approach may not be optimal because filter norms are a good measure of filter importance inside a layer, but not across layers.
- In the case of `learned_ranking` that uses re-implementation of [Learned Global Ranking method](https://arxiv.org/abs/1904.12368), a set of ranking coefficients will be learned for comparing filters across different layers.
The $(a_i, b_i)$ pair of scalars will be learned for each ( $i$ layer and used to transform norms of $i$-th layer filters before sorting all filter norms together as $a_i * N_i + b_i$ , where $N_i$ - is vector of filter norma of $i$-th layer, $(a_i, b_i)$ is ranking coefficients for $i$-th layer.
This approach allows pruning the model taking into account layer-specific sensitivity to weight perturbations and get pruned models with higher accuracy.

#### Filter pruning configuration file parameters
```
{
    "algorithm": "filter_pruning",
    "initializer": {
        "batchnorm_adaptation": {
            "num_bn_adaptation_samples": 2048, // Number of samples from the training dataset to pass through the model at initialization in order to update batchnorm statistics of the original model. The actual number of samples will be a closest multiple of the batch size.
        }
    },
    "pruning_init": 0.1, // Initial value of the pruning level applied to the convolutions that can be pruned in 'create_compressed_model' function. 0.0 by default.
    "params": {
        "schedule": "exponential", // The type of scheduling to use for adjusting the target pruning level. Either `exponential`, `exponential_with_bias`,  or `baseline`, by default it is `exponential`"
        "pruning_target": 0.4, // Target value of the pruning level for the convolutions that can be pruned. These convolutions are determined by the model architecture. 0.5 by default.
        "pruning_flops_target": 0.4, // Target value of the pruning level by FLOPs in the whole model. Only one parameter from `pruning_target` and `pruning_flops_target` can be set. If none of them is specified, `pruning_target` = 0.5 is used as the default value. 
        "num_init_steps": 3, // Number of epochs for model pretraining before starting filter pruning. 0 by default.
        "pruning_steps": 10, // Number of epochs during which the pruning level is increased from `pruning_init` to `pruning_target` value.
        "filter_importance": "L2", // The type of filter importance metric. Can be one of `L1`, `L2`, `geometric_median`. `L2` by default.
        "interlayer_ranking_type": "unweighted_ranking", // The type of filter ranking across the layers. Can be one of `unweighted_ranking`, `learned_ranking`. `unweighted_ranking` by default.
        "all_weights": false, // Whether to prune layers independently (choose filters with the smallest importance in each layer separately) or not. `False` by default.
        "prune_first_conv": false, // Whether to prune first Convolutional layers or not. First means that it is a convolutional layer such that there is a path from model input to this layer such that there are no other convolution operations on it. `False` by default (`True` by default in case of 'learned_ranking' interlayer_ranking_type).
        "prune_downsample_convs": false, // Whether to prune downsample Convolutional layers (with stride > 1) or not. `False` by default (`True` by default in case of 'learned_ranking' interlayer_ranking_type).
        "prune_batch_norms": true, // Whether to nullifies parameters of Batch Norm layer corresponds to zeroed filters of convolution corresponding to this Batch Norm. `True` by default.
        "save_ranking_coeffs_path": "path/coeffs.json", // Path to save .json file with interlayer ranking coefficients.
        "load_ranking_coeffs_path": "PATH/learned_coeffs.json", // Path to loading interlayer ranking coefficients .json file, pretrained earlier.
        "legr_params": { // Set of parameters, that can be set for 'learned_ranking' interlayer_ranking_type case
            "generations": 200, //  Number of generations for evolution algorithm optimizing. 400 by default
            "train_steps": 150, // Number of training steps to estimate pruned model accuracy. 200 by default 
            "max_pruning": 0.6, // Pruning level for the model to train LeGR algorithm on it. If learned ranking will be used for multiple pruning levels, the highest should be used as `max_pruning`. If model will be pruned with one pruning level, target pruning level should be used.
            "random_seed": 42, // Random seed for ranking coefficients generation during optimization 
        },
    },

    // A list of model control flow graph node scopes to be ignored for this operation - functions as a 'denylist'. Optional.
    "ignored_scopes": []

    // A list of model control flow graph node scopes to be considered for this operation - functions as a 'allowlist'. Optional.
    // "target_scopes": []
}
```

> **NOTE:**  In all our pruning experiments we used SGD optimizer.

#### Filter pruning statistics
A model compression can be measured by two main metrics: filter pruning level and FLOPs pruning level. While 
filter pruning level shows the ratio of removed filters to the total number of filters in the model, FLOPs pruning level 
indicates how the removed filters affect the number of floating point operations required to run a model. 

During the algorithm execution several compression statistics are available. See the example below.
```
Statistics by pruned layers:
+----------------------+------------------+--------------+---------------------+
|     Layer's name     |  Weight's shape  | Mask's shape |   Filter pruning    |
|                      |                  |              |        level        |
+======================+==================+==============+=====================+
| ConvBlock[conv1]/NNC | [192, 32, 1, 1]  | [192]        | 0.500               |
| FConv2d[conv]        |                  |              |                     |
+----------------------+------------------+--------------+---------------------+
| ConvBlock[conv2]/NNC | [384, 64, 1, 1]  | [384]        | 0.500               |
| FConv2d[conv]        |                  |              |                     |
+----------------------+------------------+--------------+---------------------+
Statistics of the pruned model:
+---------+-------+---------+---------------+
|    #    | Full  | Current | Pruning level |
+=========+=======+=========+===============+
| GFLOPS  | 0.602 | 0.241   | 0.599         |
+---------+-------+---------+---------------+
| MParams | 3.470 | 1.997   | 0.424         |
+---------+-------+---------+---------------+
| Filters | 17056 | 10216   | 0.401         |
+---------+-------+---------+---------------+
Prompt: statistic pruning level = 1 - statistic current / statistic full.
Statistics of the filter pruning algorithm:
+---------------------------------------+-------+
|           Statistic's name            | Value |
+=======================================+=======+
| Filter pruning level in current epoch | 0.500 |
+---------------------------------------+-------+
| Target filter pruning level           | 0.800 |
+---------------------------------------+-------+
```

##### Layer statistics
`Statistics by pruned layers` section lists names of all layers that will be pruned, shapes of their weight tensors, 
shapes of pruning masks applied to respective weights and percentage of zeros in those masks. 

##### Model statistics
The columns `Full` and `Current` represent the values of the corresponding statistics in the original model and compressed one in the current state, respectively.  

The `Pruning level` column indicates the ratio between the values of the full and current statistics in the corresponding rows, defined by the formula:

$Statistic\\:pruning\\:level = 1 - statistic\\:current / statistic\\:full$
  
`Filter pruning level` - percentage of filters removed from the model.  

`GFLOPs pruning level` - an estimated reduction in the number of floating point operations of the model.   
The number of FLOPs for a single convolutional layer can be calculated as:

$FLOPs = 2 * input\\:channels * kernel\\:size ^2 * W * H * filters$

> **NOTE**: One GFLOP is one billion (1e9) FLOPs.

Each removed filter contributes to FLOPs reduction in two convolutional layers as it affects the number 
of filters in one and the number of input channels of the next layer. Thus, it is expected that this number may differ 
significantly from the filter pruning level.

In addition, the decrease in GFLOPs is estimated by calculating the number of FLOPs of convolutional and fully connected layers. 
As a result, these estimates may differ slightly from the actual number of FLOPs in the compressed model.

`MParams  pruning level` - calculated reduction in the number of parameters in the model in millions. Typically convolutional layer weights have the shape of $(kernel\\:size,\\:kernel\\:size,\\:input\\:channels,\\:filter\\:num)$.

Thus, each removed filter affects the number of parameters in two convolutional layers as it affects the number 
of filters in one and the number of input channels of the next layer. It is expected that this number may differ 
significantly from the filter pruning level.

##### Algorithm statistics

`Filter (or FLOPs) pruning level in current epoch` - a pruning level calculated by the algorithm scheduler to be applied in the current training epoch. 
> **NOTE**: In case of `Filter pruning level in current epoch` this metric does not indicate the whole model filter pruning level, as 
it does not take into account the number of filters in layers that cannot be pruned.

`Target filter (or FLOPs) pruning level` - a pruning level that is expected to be achieved at the end of the algorithm execution.
> **NOTE**: In case of `Target filter pruning level` this number indicates what percentage of filters will be removed from only those layers that can be pruned.

It is important to note that pruning levels mentioned in the `statistics of the filter pruning algorithm` are the goals the algorithm aims to achieve.
It is not always possible to achieve these levels of pruning due to cross-layer and inference constraints. 
Therefore, it is expected that these numbers may differ from the calculated statistics in the `statistics of the pruned model` section.
