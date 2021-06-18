# Use Neural Network Compression Framework (NNCF) as Standalone

This is a step-by-step tutorial on how to integrate the NNCF package into the existing project.
The use case implies that the user already has a training pipeline that reproduces training of the model in the floating  point precision and pretrained model.
The task is to prepare this model for accelerated inference by simulating the compression at train time.
The instructions below use certain "helper" functions of the NNCF which abstract away most of the framework specifics and make the integration easier in most cases.
As an alternative, you can always use the NNCF internal objects and methods as described in the [architectural overview](./NNCFArchitecture.md).


## Basic usage

#### Step 1: Create an NNCF configuration file

A JSON configuration file is used for easier setup of the parameters of compression to be applied to your model.
See [configuration file description](./ConfigFile.md) or the sample configuration files packaged with the [example scripts](../examples) for reference.

#### Step 2: Modify the training pipeline
NNCF enables compression-aware training by being integrated into the regular training pipelines.
The framework is designed so that the modifications to your original training code are minor.

 1. **Add** the imports required for NNCF:
    ```python
    import torch
    import nncf  # Important - should be imported directly after torch
    from nncf import NNCFConfig, create_compressed_model, load_state
    ```
 2. Load the NNCF JSON configuration file that you prepared during Step 1:
    ```python
    nncf_config = NNCFConfig.from_json("nncf_config.json")  # Specify a path to your own NNCF configuration file in place of "nncf_config.json"
    ```
 3. (Optional) For certain algorithms such as quantization it is highly recommended to **initialize the algorithm** by
 passing training data via `nncf_config` prior to starting the compression fine-tuning properly:
    ```python
    from nncf import register_default_init_args
    nncf_config = register_default_init_args(nncf_config, train_loader, criterion)
    ```
    Training data loaders should be attached to the NNCFConfig object as part of a library-defined structure. `register_default_init_args` is a helper
    method that registers the necessary structures for all available initializations (currently quantizer range and precision initialization) by taking
    data loader, criterion and criterion function (for sophisticated calculation of loss different from direct call of the
    criterion with 2 arguments: model outputs and targets).

    The initialization expects that the model is called with its first argument equal to the dataloader output.
    If your model has more complex input arguments you can create your own data loader implementing 
    `nncf.common.initialization.dataloader.NNCFDataLoader` interface to return a tuple of (_single model input_ , _the rest of the model inputs as a kwargs dict_).

 4. Right after you create an instance of the original model and load its weights, **wrap the model** by making the following call
    ```python
    compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)
    ```
    The `create_compressed_model` function parses the loaded configuration file and returns two objects. `compression_ctrl` is a "controller" object that can be used during compressed model training to adjust certain parameters of the compression algorithm (according to a scheduler, for instance), or to gather statistics related to your compression algorithm (such as the current level of sparsity in your model).

 5. (Optional) Wrap your model with `DataParallel` or `DistributedDataParallel` classes for multi-GPU training.
If you use `DistributedDataParallel`, add the following call afterwards:
   ```python
   compression_ctrl.distributed()
   ```

   in case the compression algorithms that you use need special adjustments to function in the distributed mode.


6. In the **training loop**, make the following changes:
     - After inferring the model, take a compression loss and add it (using the `+` operator) to the common loss, for example cross-entropy loss:
        ```python
        compression_loss = compression_ctrl.loss()
        loss = cross_entropy_loss + compression_loss
        ```
     - Call the scheduler `step()` before each training iteration:
        ```python
        compression_ctrl.scheduler.step()
        ```
     - Call the scheduler `epoch_step()` before each training epoch:
        ```python
        compression_ctrl.scheduler.epoch_step()
        ```

> **NOTE**: For a real-world example of how these changes should be introduced, take a look at the [examples](../examples) published in the NNCF repository.

#### Step 3: Run the training pipeline
At this point, the NNCF is fully integrated into your training pipeline.
You can run it as usual and monitor your original model's metrics and/or compression algorithm metrics and balance model metrics quality vs. level of compression.


Important points you should consider when training your networks with compression algorithms:
  - Turn off the `Dropout` layers (and similar ones like `DropConnect`) when training a network with quantization or sparsity
  - It is better to turn off additional regularization in the loss function (for example, L2 regularization via `weight_decay`) when training the network with RB sparsity, since it already imposes an L0 regularization term.

#### Step 4 (optional): Export the compressed model to ONNX
After the compressed model has been fine-tuned to acceptable accuracy and compression stages, you can export it to ONNX format.
Since export process is in general algorithm-specific, you have to call the compression controller's `export_model` method to properly export the model with compression specifics into ONNX:
```python
compression_ctrl.export_model("./compressed_model.onnx")
```
The exported ONNX file may contain special, non-ONNX-standard operations and layers to leverage full compressed/low-precision potential of the OpenVINO toolkit.
In some cases it is possible to export a compressed model with ONNX standard operations only (so that it can be run using `onnxruntime`, for example) - this is the case for the 8-bit symmetric quantization and sparsity/filter pruning algorithms.
Refer to [compression algorithm documentation](./compression_algorithms) for details.

## Saving and loading compressed models in PyTorch
You can save the `compressed_model` object using `torch.save` as usual.
However, keep in mind that in order to load the resulting checkpoint file the `compressed_model` object should have the
same structure with regards to PyTorch module and parameters as it was when the checkpoint was saved.
In practice this means that you should use the same compression algorithms (i.e. the same NNCF configuration file) when loading a compressed model checkpoint.
Use the optional `resuming_checkpoint` argument of the `create_compressed_model` helper function to specify a PyTorch state dict to be loaded into your model once it is created.

Alternatively, you can use the `nncf.load_state` function.
It will attempt to load a PyTorch state dict into a model by first stripping the irrelevant prefixes, such as `module.` or `nncf_module.`, from both the checkpoint and the model layer identifiers, and then do the matching between the layers.
Depending on the value of the `is_resume` argument, it will then fail if an exact match could not be made (when `is_resume == True`), or load the matching layer parameters and print a warning listing the mismatches (when `is_resume == False`).
`is_resume=False` is most commonly used if you want to load the starting weights from an uncompressed model into a compressed model, and `is_resume=True` is used when you want to evaluate a compressed checkpoint or resume compressed checkpoint training without changing the compression algorithm parameters.

To save the best compressed checkpoint use `compression_ctrl.compression_stage()` to distinguish between 3 possible
levels of compression: `UNCOMPRESSED`, `PARTIALLY_COMPRESSED` and `FULLY_COMPRESSED`. It is useful in case of `staged` compression. Model may achieve
the best accuracy on earlier stages of compression - tuning without compression or with intermediate compression rate,
but still fully compressed model with lower accuracy should be considered as the best compressed one.
`UNCOMPRESSED` means that no compression is applied for the model, for instance, in case of stage quantization - when all
quantization are disabled, or in case of sparsity - when current sparsity rate is zero. `PARTIALLY_COMPRESSED` stands for the
compressed model which haven't reached final compression ratio yet, e.g. magnitude sparsity algorithm has learnt
masking of 30% weights out of 51% of target rate. The controller returns `FULLY_COMPRESSED` compression stage when it finished
scheduling and tuning hyper parameters of the compression algorithm, for example when rb-sparsity method sets final
target sparsity rate for the loss.

## Exploring the compressed model
After a `create_compressed_model` call, the NNCF log directory will contain visualizations of internal representations for the original, uncompressed model (`original_graph.dot`) and for the model with the compression algorithms applied (`compressed_graph.dot`).
These graphs form the basis for NNCF analyses of your model.
Below is the example of a LeNet network's `original_graph.dot` visualization:

![alt text](pics/lenet_original_graph.png)

Same model's `compressed_graph.dot` visualization for symmetric INT8 quantization:

![alt text](pics/lenet_compressed_graph.png)

Visualize these .dot files using Graphviz and browse through the visualization to validate that this representation correctly reflects your model structure.
Each node represents a single PyTorch function call - see [NNCFArchitecture.md](./NNCFArchitecture.md) section on graph tracing for details.
In case you need to exclude some parts of the model from being considered in one algorithm or another, you can use the labels of the `compressed_graph.dot` nodes (excluding the numerical ID in the beginning) and specify these (globally or per-algorithm) within the corresponding specific sections in [configuration file](./ConfigFile.md)
Regular expression matching is also possible for easier exclusion of certain node groups.
For instance, below is the same LeNet INT8 model as above, but with `"ignored_scopes": ["{re}.*RELU.*", "LeNet/NNCFConv2d[conv2]"]`:

![alt text](pics/lenet_compressed_graph_ignored.png)

Notice that all RELU operation outputs and the second convolution's weights are no longer quantized.


## Advanced usage

### Compression of custom modules
With no target model code modifications, NNCF only supports native PyTorch modules with respect to trainable parameter (weight) compressed, such as `torch.nn.Conv2d`
If your model contains a custom, non-PyTorch standard module with trainable weights that should be compressed, you can register it using the `@nncf.register_module` decorator:

```python
import nncf

@nncf.register_module(ignored_algorithms=[...])
class MyModule(torch.nn.Module):
    def __init__(self, ...):
        self.weight = torch.nn.Parameter(...)
    # ...
```

If registered module should be ignored by specific algorithms use `ignored_algorithms` parameter of decorator.

In the example above, the NNCF-compressed models that contain instances of `MyModule` will have the corresponding modules extended with functionality that will allow NNCF to quantize, sparsify or prune the `weight` parameter of `MyModule` before it takes part in `MyModule`'s `forward` calculation.
