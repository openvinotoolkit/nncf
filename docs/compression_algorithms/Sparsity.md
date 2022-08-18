### Non-Structured Sparsity
Sparsity algorithm zeros weights in Convolutional and Fully-Connected layers in a non-structured way,
so that zero values are randomly distributed inside the tensor. Most of the sparsity algorithms set the less important weights to zero but the criteria of how they do it is different. The framework contains several implementations of sparsity methods.

#### RB-Sparsity

This section describes the Regularization-Based Sparsity (RB-Sparsity) algorithm implemented in this framework. The method is based on $L_0$-regularization, with which parameters of the model tend to zero:

$||\theta||\_0 = \sum\limits_{i=0}^{|\theta|} \lbrack \theta\_i = 0 \rbrack$

We then reparametrize the network's weights as follows:

$\theta_{sparse}^{(i)} = \theta_i \cdot \epsilon_i, \quad \epsilon_i \sim \mathcal{B}(p_i)$

Here, $\mathcal{B}(p_i)$ is the Bernoulli distribution, $\epsilon_i$ may be interpreted as a binary mask that selects which weights should be zeroed. We then add the regularizing term to the objective function that encourages desired level of sparsity to our model:

$L_{sparse} = \mathbb{E}\_{\epsilon \sim P_{\epsilon}} \lbrack \frac{\sum\limits_{i=0}^{|\theta|} \epsilon_i}{|\theta|} - level \rbrack ^2 $

During training, we store and optimize $p_i$'s in the logit form:

$s_i = \sigma^{-1}(p_i) = log (\frac{p_i}{1 - p_i})$

and reparametrize the sampling of $\epsilon_i$'s as follows:

$\epsilon = \lbrack \sigma(s + \sigma^{-1}(\xi)) > \frac{1}{2} \rbrack, \quad \xi \sim \mathcal{U}(0,1)$

With this reparametrization, the probability of keeping a particular weight during the forward pass equals exactly to $\mathbb{P}( \epsilon_i = 1) = p_i$. We only sample the binary mask once per each training iteration. At test time, we only use the weights with $p_i > \frac{1}{2}$ as given by the trained importance scores $s_i$. To make the objective function differentiable, we treat threshold function $t(x) = x > c$ as a straight through estimator i.e. $\frac{d t}{dx} = 1$

The method requires a long schedule of the training process in order to minimize the accuracy drop.

> **NOTE**: The known limitation of the method is that the sparsified CNN must include Batch Normalization layers which make the training process more stable.

#### Batch-norm statistics adaptation

After the compression-related changes in the model have been committed, the statistics of the batchnorm layers
(per-channel rolling means and variances of activation tensors) can be updated by passing several batches of data
through the model before the fine-tuning starts. This allows to correct the compression-induced bias in the model
and reduce the corresponding accuracy drop even before model training. This option is common for quantization, magnitude
sparsity and filter pruning algorithms. It can be enabled by setting a non-zero value of `num_bn_adaptation_samples` in the
`batchnorm_adaptation` section of the `initializer` configuration (see example below).

**RB sparsity configuration file parameters**:

```
{
    "algorithm": "rb_sparsity",
    "sparsity_init": 0.05,// "Initial value of the sparsity level applied to the model in 'create_compressed_model' function
    "params": {
            "schedule": "multistep",  // The type of scheduling to use for adjusting the target sparsity level
            "patience": 3, // A regular patience parameter for the scheduler, as for any other standard scheduler. Specified in units of scheduler steps.
            "sparsity_target": 0.7, // Target value of the sparsity level for the model
            "sparsity_target_epoch": 3, // Index of the epoch from which the sparsity level of the model will be equal to spatsity_target value
            "sparsity_freeze_epoch": 50, // Index of the epoch from which the sparsity mask will be frozen and no longer trained
            "multistep_steps": [10, 20], // A list of scheduler steps at which to transition to the next scheduled sparsity level (multistep scheduler only).
            "multistep_sparsity_levels": [0.2, 0.5, 0.7] // Levels of sparsity to use at each step of the scheduler as specified in the 'multistep_steps' attribute. The first sparsity level will be applied immediately, so the length of this list should be larger than the length of the 'steps' by one. The last sparsity level will function as the ultimate sparsity target, overriding the "sparsity_target" setting if it is present.
    },

    // A list of model control flow graph node scopes to be ignored for this operation - functions as a 'denylist'. Optional.
    "ignored_scopes": []

    // A list of model control flow graph node scopes to be considered for this operation - functions as a 'allowlist'. Optional.
    // "target_scopes": []
}
```

> **NOTE**: In all our sparsity experiments, we used the Adam optimizer and initial learning rate `0.001` for model weights and sparsity mask.

#### Magnitude Sparsity

The magnitude sparsity method implements a naive approach that is based on the assumption that the contribution of lower weights is lower so that they can be pruned. After each training epoch the method calculates a threshold based on the current sparsity ratio and uses it to zero weights which are lower than this threshold. And here there are two options:
- Weights are used as is during the threshold calculation procedure.
- Weights are normalized before the threshold calculation.

**Magnitude sparsity configuration file parameters**:
```
{
    "algorithm": "magnitude_sparsity",
    "initializer": {
        "batchnorm_adaptation": {
            "num_bn_adaptation_samples": 2048, // Number of samples from the training dataset to pass through the model at initialization in order to update batchnorm statistics of the original model. The actual number of samples will be a closest multiple of the batch size.
        }
    }
    "sparsity_init": 0.05,// "Initial value of the sparsity level applied to the model in 'create_compressed_model' function
    "params": {
            "schedule": "multistep",  // The type of scheduling to use for adjusting the target sparsity level
            "patience": 3, // A regular patience parameter for the scheduler, as for any other standard scheduler. Specified in units of scheduler steps.
            "sparsity_target": 0.7, // Target value of the sparsity level for the model
            "sparsity_target_epoch": 3, // Index of the epoch from which the sparsity level of the model will be equal to spatsity_target value
            "sparsity_freeze_epoch": 50, // Index of the epoch from which the sparsity mask will be frozen and no longer trained
            "multistep_steps": [10, 20], // A list of scheduler steps at which to transition to the next scheduled sparsity level (multistep scheduler only).
            "multistep_sparsity_levels": [0.2, 0.5, 0.7] // Levels of sparsity to use at each step of the scheduler as specified in the 'multistep_steps' attribute. The first sparsity level will be applied immediately, so the length of this list should be larger than the length of the 'steps' by one. The last sparsity level will function as the ultimate sparsity target, overriding the "sparsity_target" setting if it is present.
    },

    // A list of model control flow graph node scopes to be ignored for this operation - functions as a 'denylist'. Optional.
    "ignored_scopes": []

    // A list of model control flow graph node scopes to be considered for this operation - functions as a 'allowlist'. Optional.
    // "target_scopes": []
}
```

#### Constant Sparsity
This special algorithm takes no additional parameters and is used when you want to load a checkpoint already trained with another sparsity algorithm and do other compression without changing the sparsity mask.

**Constant sparsity configuration file parameters**:
```
{
    "algorithm": "const_sparsity",
    // A list of model control flow graph node scopes to be ignored for this operation - functions as a 'denylist'. Optional.
    "ignored_scopes": []

    // A list of model control flow graph node scopes to be considered for this operation - functions as a 'allowlist'. Optional.
    // "target_scopes": []
}).
```
