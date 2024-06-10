# Frequently Asked Questions

Links to sections:

- [Common](#common)
- [PyTorch](#pytorch)
- [TensorFlow](#tensorflow)
- [ONNX](#onnx)

## Common

### What is NNCF for?

NNCF takes a deep learning network model object and modifies it for faster inference.
Within NNCF, the process of modification is colloquially known as compression.

Sometimes this is not possible to do without the loss of accuracy for the network without losing quality.
NNCF provides algorithms that strive for minimal or zero loss of accuracy, which can be applied, depending on the algorithm, during training, fine-tuning or post-training.

### Does the Neural Network *Compression* Framework provide *lossless compression*?

Not in the way the term "lossless compression" usually appears in literature.
Under "compression" we mean the preparation of the model for *future* efficient execution of this model in the OpenVINO Inference Engine.
Under "future" we mean that the process of compression is usually an offline, one-time step before the model is being used in production, which provides a new model object that could then be used instead of the original to run faster and take up lower memory without significantly losing accuracy.

No *compression* in the sense of archiving or entropy coding is being done during NNCF compression.

### How does your compression make inference faster?

General, well-known, literature-backed techniques of neural network inference acceleration (such as quantization, filter pruning and knowledge distillation) are applied, with Intel HW/runtime specifics in mind.

An overview of some of those can be found in the [following paper](https://arxiv.org/abs/2002.08679).

### Can I use NNCF-compressed models with runtimes other than OpenVINO Inference Engine?

While this is certainly possible in some cases, with a beneficial outcome even, we recommend NNCF as a way to get the most out of your setup based on OpenVINO Inference Engine inference.
We aim for best results on OpenVINO runtime with Intel hardware, and development-wise this is not always easy to generalize to other platforms or runtimes.
Some backends such as onnxruntime also support using OpenVINO Inference Engine as the actual executor for the inference, so NNCF-compressed models will also work there.

### Do I need OpenVINO or an Intel CPU to run NNCF?

Currently, this is not required in general.
Most NNCF backends can run compression and produce a compressed model object without OpenVINO or an Intel CPU on board of the machine. You only need OpenVINO and Intel hardware when you actually need to run inference on the compressed model, e.g. in a production scenario.

### Do I need a GPU to run NNCF?

Currently all NNCF-supported backends allow running in a CPU-only mode, and NNCF does not disturb this.
Note, however, that training-aware compression will naturally work much slower on most CPUs when compared with GPU-powered execution.

Check out the [notebooks](https://github.com/openvinotoolkit/openvino_notebooks#-model-training) for examples of NNCF being applied on smaller datasets which work in a reasonable amount of time on a CPU-only setup.

### NNCF supports both training and post-training compression approaches, how do I know which I need?

The rule of thumb is - start with post-training compression, and use training compression if you are not satisfied with the results and if training compression is possible for your use case.
Post-training is faster, but can degrade accuracy more than the training-enabled approach.

### I don't see any improvements after applying the `*_sparsity` algorithms

The sparsity algorithms introduce unstructured sparsity which can only be taken advantage of in terms of performance by using specialized hardware and/or software runtimes. Within the scope of these algorithms, NNCF provides functionally correct models with non-salient weights simply zeroed out, which does not lead to the reduction of the model checkpoint size. The models can, however, be used for benchmarking experimental/future hardware or runtimes, and for SOTA claims of applying unstructured sparsity on a given model architecture.

For an opportunity to observably increase performance by omitting unnecessary computations in the model, consider using the [filter pruning](./usage/training_time_compression/other_algorithms/Pruning.md) algorithm. Models compressed with this algorithm can be executed more efficiently within OpenVINO Inference Engine runtime when compared to the uncompressed counterparts.

### What is a "saturation issue" and how to avoid it?

On older generations of Intel CPUs (those not supporting [AVX-VNNI](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-VNNI)) convolutions and linear layer INT8 execution is implemented in OpenVINO's Inference Engine in such a way that mathematical overflow manifests itself *if more than 128 levels are used in the quantized domain* (out of possible 2^8 = 256) for the weights of the corresponding operations.
This is referred to as "saturation issue" within NNCF.
On newer AVX-VNNI enabled Intel CPUs the Inference Engine uses a better set of instructions that do not exhibit this flaw.

For this reason, and to support best compatibility across Intel CPUs, the weights of the convolutional and linear operations in the DL models are quantized by NNCF to only utilize 128 levels out of possible 256, effectively applying 7-bit quantization.
This does not apply to activation quantization - values of quantized activation tensors will utilize the entire set of quantization levels (256) available for INT8.

You can influence this behaviour by setting the `"overflow_fix"` parameter in the NNCF configuration file.
See documentation for this parameter in the [NNCF configuration file JSON schema reference](https://openvinotoolkit.github.io/nncf/#compression_oneOf_i0_oneOf_i0_overflow_fix).

### How can I exclude certain layers from compression?

Utilize the "ignored_scopes" parameter, either using an [NNCF config file](./ConfigFile.md) or by passing these as a function parameter if you are using NNCF purely by its Python API.
Within this parameter you can set up one or multiple identifiers to layers in your model (regex is possible) and these will be correspondingly ignored while applying the algorithms.
This can be done either globally or on a per-algorithm basis.

The format of the layer identifiers is different for each backend that NNCF supports, but attempts to be as close to the identifiers encountered in the original framework as possible.
For better understanding of how NNCF sees the layers in your network so that you can set up a working "ignored_scopes" line, see the `original_graph.dot` and the `compressed_graph.dot` Graphviz-format visualizations of the model's control flow graph.
These files are dumped in the NNCF's log directory at each invocation of model compression.

### Why do I need to pass a dataloader to certain NNCF algorithms?

These algorithms have to run a forward pass on the model to be compressed in order to properly initialize the compressed state of the model and/or to gather activation statistics that are indisposable in this algorithm.
It is recommended, although by no means mandatory, to pass a dataloader with the same dataset that you were training the initial model for.

### The compression process takes too long, how can I make it faster?

For training approaches the majority of time is taken by the training loop, so any regular methods that improve model convergence should work here.
Try the built-in [knowledge distillation](./usage/training_time_compression/other_algorithms/KnowledgeDistillation.md) to potentially obtain target accuracy faster.
Alternatively you may want to reduce the number of initialization samples taken from the initialization dataloader by the algorithms that require it.

### I get a "CUDA out of memory" error when running NNCF in the compression-aware training approach, although the original model to be compressed runs and trains fine without NNCF

As some of the compression algorithm parameters are also trainable, NNCF-compressed model objects ready for training will have a larger GPU memory footprint than the uncompressed counterparts.
Try reducing batch size for the NNCF training runs if it makes sense to do so in your situation.

## PyTorch

### Importing anything from `nncf.torch` hangs

NNCF utilizes the [torch C++ extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html) mechanism to accelerate the quantization-aware training process.
This is done by just-in-time compiling a set of C++/CUDA files using the system-local compilers and toolsets.
The compilation happens at the first import of `nncf.torch` or anything under that namespace on the machine, or within the current Python environment.
The result is a set of the `.so` files containing the compiled extensions binary code stored in a system-specific location (commonly `~/.cache/torch_extensions`, or alternatively wherever `TORCH_EXTENSIONS_DIR` environment variable points to).

To avoid race conditions, PyTorch uses `lock` files in this folder during compilation.
Sometimes, when the compilation process is abnormally aborted, these `lock` files remain in the filesystem, which leads to a hang the next time you import anything from `nncf.torch`, because the just-in-time compilation process will wait indefinitely until the `lock` files have been cleared.
To resolve these, delete the `torch_extensions` directory (at `~/.cache`, or pointed to by `TORCH_EXTENSIONS_DIR`, or at your specific location), and re-run the script that imports from `nncf.torch`.
The compilation takes some time and happens upon import, so do not interrupt the launch of your Python script until the import has been completed.

### Importing anything from `nncf.torch` leads to an error mentioning `gcc`, `nvcc`, `ninja`, or `cl.exe`

See the answer above for the general description of the reasons why these are involved in NNCF PyTorch operation.
To resolve, make sure that your CUDA installation contains the development tools (e.g. the `nvcc` compiler), and that the environmental variables are set properly so that these tools are available in `PATH` or `PYTHONPATH`.

### My model trains and runs slower in PyTorch when compressed by NNCF

NNCF does not in general accelerate training or inference when the compressed model is run in PyTorch.
It only prepares the model for further inference with OpenVINO's Inference Engine, where the runtime has capabilities of processing the NNCF-compressed models so that they run faster than their uncompressed counterparts.

The process of compressing in PyTorch relies on hooking regular PyTorch functions and calling extra code for purposes of compression algorithm logic, so the NNCF-processed models will inevitably run slower in PyTorch. Export your model after processing with NNCF to an OpenVINO-ingestible format (e.g. ONNX) and run it with the OpenVINO Inference Engine, to enjoy speedups when compared to the uncompressed model inference with Inference Engine.

### The .pth checkpoints for the compressed model have larger size and parameter count when compared to the uncompressed model

See the answer to the above question. Additional parameters are part of the compression algorithm internal state being saved along with the regular model weights, and any model size footprint reduction is deferred until exporting and/or running the model with OpenVINO Inference Engine.

### My RNN model is not compressed completely or fails at the compression stage

Currently NNCF PyTorch can only properly handle models with acyclic execution graphs.
RNNs, which inherently have cycles, can behave oddly when processed with NNCF PyTorch, which includes loss of quality, unreproducible results and failure to compress.

<a name="pt_init_dataloader"></a>

### I get a `Could not deduce the forward arguments from the initializing dataloader output.` runtime error when executing `create_compressed_model`

Dataloaders can return anything, and this output may be preprocessed in the rest of the training pipeline before actually ending up in model's `forward` method.
NNCF needs a dataloader already at the compressed model creation stage, e.g. before training, and doesn't in general know about the further preprocessing (turning the output of `v8_dataloader` into actual `forward` args and kwargs.
You have to give NNCF this information by wrapping your dataloader object in an own subclass of a `nncf.torch.initialization.PTInitializingDataLoader` object that properly defines the `get_inputs` and `get_target` abstract methods:

```python
from nncf.torch.initialization import PTInitializingDataLoader

class MyInitializingDataLoader(PTInitializingDataLoader):
    def  get_inputs(self, dataloader_output: Any) -> Tuple[Tuple, Dict]:
        # your implementation - `dataloader_output` is what is returned by your dataloader,
        # and you have to turn it into a (args, kwargs) tuple that is required by your model
        # in this function, for instance, if your dataloader returns dictionaries where
        # the input image is under key `"img"`, and your YOLOv8 model accepts the input
        # images as 0-th `forward` positional arg, you would do:
        return (dataloader_output["img"],), {}

   def get_target(self, dataloader_output: Any) -> Any:
        # and in this function you should extract the "ground truth" value from your
        # dataloader, so, for instance, if your dataloader output is a dictionary where
        # ground truth images are under a "gt" key, then here you would write:
        return dataloader_output["gt"]

init_dataloader = MyInitializingDataLoader(my_dataloader)
# now you pass this wrapped object instead of your original dataloader into the `register_default_init_args`
nncf_config = register_default_init_args(nncf_config, init_dataloader)
# and then call `create_compressed_model` with that config file as usual.
```

## TensorFlow

*To be filled*

## ONNX

*To be filled*
