# Knowledge Distillation (experimental feature)

## The algorithm description

The Knowledge Distillation [Hinton et al., 2015](https://arxiv.org/pdf/1503.02531.pdf)
implies that a small model (student) is trained to mimic a pre-trained large model (teacher) through knowledge
transfer. The goal is to improve the accuracy of the student network.

The NNCF for PyTorch supports Knowledge Distillation out of the box along with all supported compression algorithm
(quantization, sparsity, filter pruning), when a student is a model being compressed and teacher - original
non-compressed one.

Knowledge is transferred from the teacher model to the student one by minimizing loss function, which is calculated
based on predictions of the models. At the moment, two types of loss functions are available.
One of them should be explicitly specified in the config.

MSE distillation loss:

${L}_{MSE}(z^{s}, z^{t}) = || z^s - z^t ||_2^2$

Cross-Entropy distillation loss:

${p}_{i} = \frac{\exp({z}\_{i})}{\sum\_{j}(\exp({z}\_{j}))}$

${L}\_{CE}({p}^{s}, {p}^{t}) = -\sum_{i}{p}^{t}\_{i}*\log({p}^{s}\_{i})$

The Knowledge Distillation loss function is combined with a regular loss function, so overall loss function will be
computed as:

 $L = {L}\_{reg}({z}^{s}, y) + {L}\_{distill}({z}^{s}, {z}^{t})$

 ![kd_pic](/docs/pics/knowledge_distillation.png)

  Note: the Cross-Entropy distillation loss was proposed in [Hinton et al., 2015](https://arxiv.org/pdf/1503.02531.pdf)
  with temperature parameter, but we don't use it or assume that T=1.

## User guide

To turn on the Knowledge Distillation with some compression algorithm (e.g. filter_pruning) it's necessary to
specify `knowledge_distillation` algorithm and its type in the config:

```json
{
    ...
    "compression": [
        {
            "algorithm": "filter_pruning",
            ...
        },
        {
            "algorithm": "knowledge_distillation",
            "type": "softmax" // or "type": "mse"
        }
    ]
}
```

See this [config file](/examples/torch/classification/configs/pruning/resnet34_imagenet_pruning_geometric_median_kd.json) for an example, and [NNCF config schema](https://openvinotoolkit.github.io/nncf/) for reference to the available configuration parameters for the algorithm.

## Limitations

- The algorithm is supported for PyTorch only.
- Training the same configuration with Knowledge Distillation requires more time and GPU memory than without it.
On average, memory (for all GPU execution modes) and time overhead is below 20% each.
- Outputs of model that shouldn't be differentiated must have `requires_grad=False`.
- Model should output predictions, not calculate the losses.
