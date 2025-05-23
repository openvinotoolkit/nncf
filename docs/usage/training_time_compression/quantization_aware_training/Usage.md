# Use NNCF for Quantization Aware Training

This is a step-by-step tutorial on how to integrate the NNCF package into the existing PyTorch or TensorFlow projects.
The use case implies that the user already has a training pipeline that reproduces training of the model in the floating point precision and pretrained model.
The task is to prepare this model for accelerated inference by simulating the compression at train time.
Please refer to this [document](/docs/usage/training_time_compression/other_algorithms/LegacyQuantization.md) for details of the implementation.

## Basic usage

### Step 1: Apply Post Training Quantization to the Model

Quantize the model using the [Post Training Quantization](../../post_training_compression/post_training_quantization/Usage.md) method.

<details open><summary><b>PyTorch</b></summary>

```python
model = TorchModel() # instance of torch.nn.Module
quantized_model = nncf.quantize(model, ...)
```

</details>

<details><summary><b>TensorFlow</b></summary>

```python
model = TensorFlowModel() # instance of tf.keras.Model
quantized_model = nncf.quantize(model, ...)
```

</details>

### Step 2: Run the training pipeline

At this point, the NNCF is fully integrated into your training pipeline.
You can run it as usual and monitor your original model's metrics and/or compression algorithm metrics and balance model metrics quality vs. level of compression.

Important points you should consider when training your networks with compression algorithms:

- Turn off the `Dropout` layers (and similar ones like `DropConnect`) when training a network with quantization

### Step 3: Export the compressed model

After the compressed model has been fine-tuned to acceptable accuracy and compression stages, you can export it.

<details open><summary><b>PyTorch</b></summary>

Trace the model via inference in framework operations.

```python
# To OpenVINO format
import openvino as ov
ov_quantized_model = ov.convert_model(quantized_model.cpu(), example_input=dummy_input)
```

</details>

<details><summary><b>TensorFlow</b></summary>

```python
# To OpenVINO format
import openvino as ov

ov_quantized_model = ov.convert_model(quantized_model)
```

</details>

## Saving and loading compressed models

<details open><summary><b>PyTorch</b></summary>

The complete information about compression is defined by a compressed model and a NNCF config.
The model characterizes the weights and topology of the network. The NNCF config - how to restore additional modules introduced by NNCF.
The NNCF config can be obtained by `nncf.torch.get_config` on saving and passed to the
`nncf.torch.load_from_config` helper function to load additional modules from the given NNCF config.
The quantized model saving allows to load quantized modules to the target model in a new python process and
requires only example input for the target module, corresponding NNCF config and the quantized model state dict.

```python
import nncf.torch

# save part
quantized_model = nncf.quantize(model, calibration_dataset)
checkpoint = {
    'state_dict': quantized_model.state_dict(),
    'nncf_config': nncf.torch.get_config(quantized_model),
    ...
}
torch.save(checkpoint, path)

# load part
resuming_checkpoint = torch.load(path)

nncf_config = resuming_checkpoint['nncf_config']
state_dict = resuming_checkpoint['state_dict']

quantized_model = nncf.torch.load_from_config(model, nncf_config, dummy_input)
quantized_model.load_state_dict(state_dict)
```

You can save the `compressed_model` object `torch.save` as usual: via `state_dict` and `load_state_dict` methods.

</details>

<details><summary><b>TensorFlow</b></summary>

To save a model checkpoint, use the following API:

```python
from nncf.tensorflow import ConfigState
from nncf.tensorflow import get_config
from nncf.tensorflow.callbacks.checkpoint_callback import CheckpointManagerCallback

nncf_config = get_config(quantized_model)
checkpoint = tf.train.Checkpoint(model=quantized_model,
                                 nncf_config_state=ConfigState(nncf_config),
                                 ... # the rest of the user-defined objects to save
                                 )
callbacks = []
callbacks.append(CheckpointManagerCallback(checkpoint, path_to_checkpoint))
...
quantized_model.fit(..., callbacks=callbacks)
```

To restore the model from checkpoint, use the following API:

```python
from nncf.tensorflow import ConfigState
from nncf.tensorflow import load_from_config

checkpoint = tf.train.Checkpoint(nncf_config_state=ConfigState())
checkpoint.restore(path_to_checkpoint)

quantized_model = load_from_config(model, checkpoint.nncf_config_state.config)

checkpoint = tf.train.Checkpoint(model=quantized_model
                                 ... # the rest of the user-defined objects to load
                                 )
checkpoint.restore(path_to_checkpoint)
```

</details>

## Advanced usage

### Compression of custom modules

<details open><summary><b>PyTorch</b></summary>

With no target model code modifications, NNCF only supports native PyTorch modules with respect to trainable parameter (weight) compressed, such as `torch.nn.Conv2d`.
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

In the example above, the NNCF-compressed models that contain instances of `MyModule` will have the corresponding modules extended with functionality that will allow NNCF to quantize the `weight` parameter of `MyModule` before it takes part in `MyModule`'s `forward` calculation.

</details>

## Examples

- See a PyTorch [example](/examples/quantization_aware_training/torch/resnet18/README.md) for **Quantization** Compression scenario on Tiny ImageNet-200 dataset.
- See a TensorFlow [example](/examples/quantization_aware_training/tensorflow/mobilenet_v2/README.md) for **Quantization** Compression scenario on imagenette/320px-v2 dataset.
