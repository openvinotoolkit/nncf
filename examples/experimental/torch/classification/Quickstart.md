# Setup

## NNCF and PyTorch

Install NNCF and PyTorch using the latest instructions in https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md.

The ```examples``` folder from the NNCF repository ***is not*** included when you install NNCF using a package manager. To run the BootstrapNAS examples, you will need to obtain this folder from the repository and add it to your path.

## Additional Dependencies

The examples in the NNCF repo have additional requirements, such as EfficientNet, MLFlow, Tensorboard, etc., which are not installed with NNCF. You will need to install them using:

```bash
pip install efficientnet_pytorch tensorboard mlflow returns
```

## Example

To run an example of super-network generation and sub-network search, use the ```bootstrap_nas.py``` script located [here](https://github.com/openvinotoolkit/nncf/blob/develop/examples/experimental/torch/classification/bootstrap_nas.py) and the sample ```config.json``` from [here](https://github.com/jpablomch/bootstrapnas/blob/main/bootstrapnas_examples/config.json).

The file ```config.json``` contains a sample configuration for generating a super-network from a trained model. The sample file is configured to generate a super-network from ResNet-50 trained with CIFAR-10. The file should be modified depending on the model to be used as input for BootstrapNAS.

Weights for CIFAR10-based models can be found at: https://github.com/huyvnphan/PyTorch_CIFAR10

Use the following to test training a super-network:

```bash
cd <path to NNCF>/examples/experimental/torch/classification
python bootstrap_nas.py -m train \
    -c <path to this repo>/bootstrapnas_examples/config.json \
    --data <path to your CIFAR10 dataset> \
    --weights <path to weights for resnet-50 trained with CIFAR10>
```

### Expected Output Files after executing BootstrapNAS

The output of running ```bootstrap_nas.py``` will be a sub-network configuration that has an accuracy similar to the input model (by default a $\pm$1% absolute difference in accuracy is allowed), but with improvements in MACs. Format: ([MACs_subnet, ACC_subnet]).

Several files are saved to your `log_dir` after the training has ended:

- `compressed_graph.{dot, png}`- Dot and PNG files that describe the wrapped NNCF model.
- `original_graph.dot` - Dot file that describes the original model.
- `config.json`- A copy of your original config file.
- `events.*`- Tensorboard logs.
- `last_elasticity.pth`- Super-network's elasticity information. This file can be used when loading super-networks for searching or inspection.
- `last_model_weights.pth`- Super-network's weights after training.
- `snapshot.tar.gz` - Copy of the code used for this run.
- `subnetwork_best.pth` - Dictionary with the configuration of the best sub-network. Best defined as a sub-network that performs in the Pareto front, and that deviates a maximum `acc_delta` from original model.
- `supernet_{best, last}.pth` - Super-network weights at its best and last state.

If the user wants to have a CSV output file of the search progression, ```search_algo.search_progression_to_csv()``` can be called after running the search step.

For a visualization of the search progression please use ```search_algo.visualize_search_progression()``` after the search has concluded. A PNG file will be generated.
