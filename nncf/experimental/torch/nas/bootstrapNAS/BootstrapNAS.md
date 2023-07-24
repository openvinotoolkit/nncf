# BootstrapNAS

Automated generation of weight-sharing super-networks (Cai, Gan, et al., 2020) for Neural Architecture Search (NAS) (Elsken et al., 2019). A weight-sharing super-network is a data structure from which smaller and more efficient sub-networks can be extracted.

<p align="center">
<img src="architecture.png" alt="BootstrapNAS Architecture" width="500"/>
</p>

BootstrapNAS (1) takes as input a pre-trained model. (2) It uses this model to generate a weight-sharing super-network. (3) BootstrapNAS then applies a training strategy, and once the super-network has been trained, (4) it searches for efficient subnetworks that satisfy the user's requirements. (5) The configuration of the discovered sub-network(s) is returned to the user.

The parameters for generating, training and searching on the super-network are defined in a configuration file within two exclusive subsets of parameters for training and search:

```json5
"bootstrapNAS": {
    "training": {
        ...
    },
    "search": {
        ...
    }
}
```

In the `training` section, you specify the training algorithm, e.g., `progressive_shrinking`, schedule and elasticity parameters:

```json
"training": {
    "algorithm": "progressive_shrinking",
    "progressivity_of_elasticity": ["depth", "width"],
    "batchnorm_adaptation": {
        "num_bn_adaptation_samples": 1500
    },
    "schedule": {
        "list_stage_descriptions": [
            {"train_dims": ["depth"], "epochs": 25, "depth_indicator": 1, "init_lr": 2.5e-6, "epochs_lr": 25},
            {"train_dims": ["depth"], "epochs": 40, "depth_indicator": 2, "init_lr": 2.5e-6, "epochs_lr": 40},
            {"train_dims": ["depth", "width"], "epochs": 50, "depth_indicator": 2, "reorg_weights": true, "width_indicator": 2, "bn_adapt": true, "init_lr": 2.5e-6, "epochs_lr": 50},
            {"train_dims": ["depth", "width"], "epochs": 50, "depth_indicator": 2, "reorg_weights": true, "width_indicator": 3, "bn_adapt": true, "init_lr": 2.5e-6, "epochs_lr": 50}
        ]
    },
    "elasticity": {
        "available_elasticity_dims": ["width", "depth"],
        "width": {
            "max_num_widths": 3,
            "min_width": 32,
            "width_step": 32,
            "width_multipliers": [1, 0.80, 0.60]
    },
    ...
}
```

In the search section, you specify the search algorithm, e.g., `NSGA-II` and its parameters. For example:

```json
"search": {
    "algorithm": "NSGA2",
    "num_evals": 3000,
    "population": 50,
    "ref_acc": 93.65
}
```

By default, BootstrapNAS uses `NSGA-II` (Dev et al., 2002), an genetic algorithm that constructs a pareto front of efficient sub-networks.

List of parameters that can be used in the configuration file:

**Training:**

`algorithm`: Defines training strategy for tuning supernet. By default, `progressive_shrinking`.

`progressivity_of_elasticity`: Defines the order of adding a new elasticity dimension from stage to stage.
examples=["width", "depth", "kernel"].

`batchnorm_adaptation`: Specifies the number of samples from the training dataset to use for model inference during the
BatchNorm statistics adaptation procedure for the compressed model.

`schedule`: The schedule section includes a list of stage descriptors (`list_stage_descriptions`) that specify the
elasticity dimensions enabled for a particular stage (`train_dims`), the number of `epochs` for the stage, the
`depth_indicator` which in the case of elastic depth, restricts the maximum number of blocks in each independent group
that can be skipped, the `width_indicator`, which restricts the maximum number of width values in each elastic layer.
The user can also specify whether weights should be reorganized (`reorg_weights`), whether batch norm adaptation should
be triggered at the beginning of the stage (`bn_adapt`), the initial learning rate for the stage (`init_lr`), and
the epochs to use for adjusting the learning rate (`epochs_lr`).

`elasticity`: Currently, BootstrapNAS supports three elastic dimensions (`kernel`, `width` and `depth`).
Elastic depth automatically finds blocks to skip, by default. The user can specify the `min_block_size`, i.e., minimal
number of operations in the skipping block, and the `max_block_size`, i.e., maximal number of operations in the block.
Alternatively, one can specify list of blocks to skip manually via `skipped_blocks`.
In the case of elastic width, the user can specify the `min_width`, i.e., the minimal number of output channels that
can be activated for each layers with elastic width. Default value is 32, the `max_num_widths`, which restricts total
number of different elastic width values for each layer, a `width_step`, which defines a step size for a generation of
the elastic width search space, or a `width_multiplier` to define the elastic width search space via a list of multipliers.
Finally, the user can determine the type of filter importance metric: L1, L2 or geometric mean. L2 is selected by default.
For elastic kernel, the user can specify the `max_num_kernels`, which restricts the total number of different elastic
kernel values for each layer.

`train_steps`: Defines the number of samples used for each training epoch.

**Search:**

`algorithm`: Defines the search algorithm. The default algorithm is NSGA-II.

`num_evals`: Defines the number of evaluations that will be used by the search algorithm.

`population`: Defines the population size when using an evolutionary search algorithm.

`acc_delta`: Defines the absolute difference in accuracy that is tolerated when looking for a subnetwork.

`ref_acc`: Defines the reference accuracy from the pre-trained model used to generate the super-network.

For more information about BootstrapNAS and to cite this work, please refer to the following publications:

[Automated Super-Network Generation for Scalable Neural Architecture Search](https://openreview.net/attachment?id=HK-zmbTB8gq&name=main_paper_and_supplementary_material).

```bibtex
  @inproceedings{
    munoz2022automated,
    title={Automated Super-Network Generation for Scalable Neural Architecture Search},
    author={Mu{\~{n}}oz, J. Pablo and Lyalyushkin, Nikolay and Lacewell, Chaunte and Senina, Anastasia and Cummings, Daniel and Sarah, Anthony  and Kozlov, Alexander and Jain, Nilesh},
    booktitle={First Conference on Automated Machine Learning (Main Track)},
    year={2022},
    url={https://openreview.net/forum?id=HK-zmbTB8gq}
  }
```

[Enabling NAS with Automated Super-Network Generation](https://arxiv.org/abs/2112.10878)

```bibtex
@article{
  bootstrapNAS,
  author    = {Mu{\~{n}}oz, J. Pablo  and Lyalyushkin, Nikolay  and Akhauri, Yash and Senina, Anastasia and Kozlov, Alexander  and Jain, Nilesh},
  title     = {Enabling NAS with Automated Super-Network Generation},
  journal   = {CoRR},
  volume    = {abs/2112.10878},
  year      = {2021},
  url       = {https://arxiv.org/abs/2112.10878},
  eprinttype = {arXiv},
  eprint    = {2112.10878},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2112-10878.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## References

- Cai, H., C. Gan, et al. (2020). “Once for All: Train One Network and Specialize it for Efficient
Deployment”. In: International Conference on Learning Representations.
- Deb, K., A. Pratap, et al. (2002). “A fast and elitist multiobjective genetic algorithm: NSGA-II”. In: 303
IEEE Transactions on Evolutionary Computation 6.2, pp. 182–197.
- Elsken, T., J. H. Metzen, and F. Hutter (2019). “Neural Architecture Search: A Survey”. In: Journal of  Machine Learning Research 20.55, pp. 1–21.
