# Use Neural Network Compression Framework (NNCF) for Quantization Aware Training

This is a step-by-step tutorial on how to integrate the NNCF package into the existing project.
The use case implies that the user already has a training pipeline that reproduces training of the model in the floating  point precision and pretrained model.
The task is to prepare this model for accelerated inference by simulating the compression at train time.
The instructions below use certain "helper" functions of the NNCF which abstract away most of the framework specifics and make the integration easier in most cases.
As an alternative, you can always use the NNCF internal objects and methods as described in the [architectural overview](./NNCFArchitecture.md).

## Basic usage

### Step 1: Modify the training pipeline

NNCF enables compression-aware training by being integrated into the regular training pipelines.
The framework is designed so that the modifications to your original training code are minor.

 1. **Add** the imports required for NNCF:

    ```python
    import torch
    import nncf.torch  # Important - must be imported before any other external package that depends on torch
    ```

    **NOTE (PyTorch)**: Due to the way NNCF works within the PyTorch backend, `import nncf` must be done before any other import of `torch` in your package _or_ in third-party packages that your code utilizes, otherwise the compression may be applied incompletely.

 2. Create the data transformation function

    ```python
    def transform_fn(data_item):
        images, _ = data_item
        return images

    calibration_dataset = nncf.Dataset(train_loader, transform_fn) # train_loader is an instance of torch.utils.data.DataLoader
    ```

 3. Right after you create an instance of the original model and load its weights, **wrap the model** by making the following call

    ```python
    quantized_model = nncf.quantize(model, calibration_dataset)
    ```

### Step 3: Run the training pipeline

At this point, the NNCF is fully integrated into your training pipeline.
You can run it as usual and monitor your original model's metrics and/or compression algorithm metrics and balance model metrics quality vs. level of compression.

Important points you should consider when training your networks with compression algorithms:

- Turn off the `Dropout` layers (and similar ones like `DropConnect`) when training a network with quantization or sparsity
- It is better to turn off additional regularization in the loss function (for example, L2 regularization via `weight_decay`) when training the network with RB sparsity, since it already imposes an L0 regularization term.

### Step 4: Export the compressed model

After the compressed model has been fine-tuned to acceptable accuracy and compression stages, you can export it. There are two ways to export a model:

1. Trace the model via inference in framework operations.

    ```python
    # To ONNX format
    import torch
    torch.onnx.export(quantized_model, dummy_input, './compressed_model.onnx')
    # To OpenVINO format
    import openvino as ov
    ov_quantized_model = ov.convert_model(quantized_model.cpu(), example_input=dummy_input)
    ```

## Saving and loading compressed models

The complete information about compression is defined by a compressed model and a NNCF config.
The model characterizes the weights and topology of the network. The NNCF config - how to restore additional modules intoduced by NNCF.
The NNCF config can be obtained by `quantized_model.nncf.get_config()` on saving and passed to the
`nncf.torch.load_from_config` helper function to loading additional modules from the given NNCF config.

### Saving and loading compressed models in PyTorch

```python
# save part
quantized_model = nncf.quantize(model, calibration_dataset)
checkpoint = {
    'state_dict':quantized_model.state_dict(),
    'nncf_config': quantized_model.nncf.get_config(),
    ...
}
torch.save(checkpoint, path)

# load part
resuming_checkpoint = torch.load(path)
nncf_config = resuming_checkpoint['nncf_config']
quantized_model = nncf.torch.load_from_config(model, nncf_config, dummy_input)
state_dict = resuming_checkpoint['state_dict']
quantized_model.load_state_dict(state_dict)
```

You can save the `compressed_model` object `torch.save` as usual: via `state_dict` and `load_state_dict` methods.

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

See a PyTorch [example](examples/quantization_aware_training/torch/resnet18/main.py) for **Quantization** Compression scenario on Tiny ImageNet-200 dataset.
