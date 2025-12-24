# Pruning

## Non-Structured Pruning

Sparsity algorithm zeros weights in a non-structured way, so that zero values are randomly distributed inside
the tensor. Most of the pruning algorithms set the less important weights to zero but the criteria of how they
do it is different. The framework contains several implementations of pruning methods.

### Magnitude

The magnitude pruning method implements a naive approach that is based on the assumption that the contribution
of lower weights is lower so that they can be pruned. After each training epoch the method calculates a threshold based
on the current pruning ratio and uses it to zero weights which are lower than this threshold.

And here there are two options:

* `UNSTRUCTURED_MAGNITUDE_LOCAL`: Unstructured magnitude-based pruning with local importance calculation.
    Weight importance is computed independently for each tensor.

* `UNSTRUCTURED_MAGNITUDE_GLOBAL`: Unstructured magnitude-based pruning with global importance calculation.
    Weight importance is computed across all tensors selected for pruning.

```python
import nncf

...

pruned_model = nncf.prune(
    model,
    mode=nncf.PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL,
    ratio=0.7,
    examples_inputs=example_input,
)
```

To get a more accurate model, it is recommended to fine-tune the model for several epochs or use batch norm adaptation.

#### Batch Norm Adaptation after Pruning

When using magnitude pruning without fine-tuning, it is recommended to perform Batch Norm adaptation after pruning to get more accurate results.

```python
import nncf

...

def transform_fn(batch: tuple[torch.Tensor, int]) -> torch.Tensor:
    inputs, _ = batch
    return inputs.to(device=device)

calibration_dataset = nncf.Dataset(train_loader, transform_func=transform_fn)

pruned_model = nncf.batch_norm_adaptation(
    pruned_model,
    calibration_dataset=calibration_dataset,
    num_iterations=200,
)
```

## Regularization-Based

The method is based on $L_0$-regularization, with which parameters of the model tend to zero:

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

```python
import nncf
from nncf.torch.function_hook.pruning.rb.losses import RBLoss

...

pruned_model = nncf.prune(
    model,
    mode=nncf.PruneMode.UNSTRUCTURED_REGULARIZATION_BASED,
    examples_inputs=example_input,
)
num_epochs = 30
rb_loss = RBLoss(pruned_model, target_ratio=0.7, p=0.1).to(device)

...

for epoch in range(num_epochs):
    for batch in train_loader:
        ...
        outputs = pruned_model(inputs)
        task_loss = criterion(outputs, targets)
        reg_loss = rb_loss()
        loss = task_loss + reg_loss
```

## Statistics

To gather statistics about the pruning process, use the following code:

```python
stat = nncf.pruning_statistic(pruned_model)
print(stat)
```

> [!NOTE]
> Statistics about the pruning process cannot be gathered after using `nncf.strip`

## Strip

The strip function is used to permanently apply the pruning masks to the
model weights and to remove all auxiliary pruning-related operations.

After calling this function, the masks are merged into the weights,
and any additional layers, parameters, or forward-pass operations
introduced for pruning are removed. The resulting model contains only
the pruned weights and can be used for inference without pruning overhead.

```python
nncf.strip(pruned_model, strip_format=nncf.StripFormat.IN_PLACE)
```
