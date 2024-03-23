# Non-Structured Sparsity

>_Scroll down for the examples of the JSON configuration files that can be used to apply this algorithm_.

Sparsity algorithm zeros weights in Convolutional and Fully-Connected layers in a non-structured way,
so that zero values are randomly distributed inside the tensor. Most of the sparsity algorithms set the less important weights to zero but the criteria of how they do it is different. The framework contains several implementations of sparsity methods.

## RB-Sparsity

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

## Batch-norm statistics adaptation

After the compression-related changes in the model have been committed, the statistics of the batchnorm layers
(per-channel rolling means and variances of activation tensors) can be updated by passing several batches of data
through the model before the fine-tuning starts. This allows to correct the compression-induced bias in the model
and reduce the corresponding accuracy drop even before model training. This option is common for quantization, magnitude
sparsity and filter pruning algorithms. It can be enabled by setting a non-zero value of `num_bn_adaptation_samples` in the
`batchnorm_adaptation` section of the `initializer` configuration (see example below).

> **NOTE**: In all our sparsity experiments, we used the Adam optimizer and initial learning rate `0.001` for model weights and sparsity mask.

## Magnitude Sparsity

The magnitude sparsity method implements a naive approach that is based on the assumption that the contribution of lower weights is lower so that they can be pruned. After each training epoch the method calculates a threshold based on the current sparsity ratio and uses it to zero weights which are lower than this threshold. And here there are two options:

- Weights are used as is during the threshold calculation procedure.
- Weights are normalized before the threshold calculation.

## Constant Sparsity

This special algorithm takes no additional parameters and is used when you want to load a checkpoint already trained with another sparsity algorithm and do other compression without changing the sparsity mask.

### Example configuration files

>_For the full list of the algorithm configuration parameters via config file, see the corresponding section in the [NNCF config schema](https://openvinotoolkit.github.io/nncf/)_.

- Apply magnitude sparsity with default parameters (0 to 90% sparsity over 90 epochs of training, sparsity increased polynomially with each epoch):

```json5
{
    "input_info": { "sample_size": [1, 3, 224, 224] }, // the input shape of your model may vary
    "compression": {
      "algorithm": "magnitude_sparsity"
    }
}
```

- Apply magnitude sparsity, increasing sparsity level step-wise from 0 to 70% in 3 steps at given training epoch indices:

```json5
{
    "input_info": { "sample_size": [1, 3, 224, 224] }, // the input shape of your model may vary
    "compression": {
      "algorithm": "magnitude_sparsity",
      "params": {
        "schedule": "multistep",
        "multistep_steps": [10, 20],
        "multistep_sparsity_levels": [0, 0.35, 0.7], // first level applied immediately (epoch 0), 0.35 - at epoch 10, 0.7 - at epoch 20
        "sparsity_target": 0.5,
        "sparsity_target_epoch": 20 // "sparsity_target" fully reached at the beginning of epoch 20
      }
    }
}
```

- Apply magnitude sparsity, immediately setting sparsity level to 10%, performing [batch-norm adaptation](./BatchnormAdaptation.md) to potentially recover accuracy, and exponentially increasing sparsity to 50% over 30 epochs of training:

```json5
{
    "input_info": { "sample_size": [1, 3, 224, 224] }, // the input shape of your model may vary
    "compression": {
      "algorithm": "magnitude_sparsity",
      "sparsity_init": 0.1,  // set already before the beginning of epoch 0 of training
      "params": {
        "schedule": "exponential",
        "sparsity_target": 0.5,
        "sparsity_target_epoch": 30 // "sparsity_target" fully reached at the beginning of epoch 20
      },
      "initializer": {
        "batchnorm_adaptation": {
          "num_bn_adaptation_samples": 100
        }
      }
    }
}
```

- Apply RB-sparsity to UNet, increasing sparsity level exponentially from 1% to 60% over 100 epochs, keeping the sparsity mask trainable until epoch 110 (after which the mask is frozen and the model is allowed to fine-tune with a fixed sparsity level), and excluding parts of the model from sparsification:

```json5
{
    "input_info": { "sample_size": [1, 3, 224, 224] }, // the input shape of your model may vary
    "compression": {
        "algorithm": "rb_sparsity",
        "sparsity_init": 0.01,
        "params": {
            "sparsity_target": 0.60,
            "sparsity_target_epoch": 100,
            "sparsity_freeze_epoch": 110
        },
        "ignored_scopes": [
          // assuming PyTorch model
           "{re}UNet/ModuleList\\[up_path\\].*", // do not sparsify decoder
           "UNet/NNCFConv2d[last]/conv2d_0" // do not sparsify final convolution
        ]
    }
}
```
