# Frequently Asked Questions

Links to sections:
- [Common](#common)
- [PyTorch](#pytorch)
- [TensorFlow](#tensorflow)
- [ONNX](#onnx)


## Common

### Does the Neural Network *Compression* Framework provide *lossless compression*?
Not in the way the term "lossless compression" usually appears in literature. Under "compression" we mean the preparation of the model for *future* efficient execution of this model in the OpenVINO Inference Engine. Under "future" we mean that the process of compression is usually an offline, one-time step before the model is being used in production, which provides a new model object that could then be used instead of the original to run faster and take up lower memory without significantly losing accuracy.

No *compression* in the sense of archiving or entropy coding is being done during NNCF compression.

### I don't see any improvements after applying the `*_sparsity` algorithms
The sparsity algorithms introduce unstructured sparsity which can only be taken advantage of in terms of performance by using specialized hardware and/or software runtimes. Within the scope of these algorithms, NNCF provides functionally correct models with non-salient weights simply zeroed out, which does not lead to the reduction of the model checkpoint size. The models can, however, be used for benchmarking experimental/future hardware or runtimes, and for SOTA claims of applying unstructured sparsity on a given model architecture.

For an opportunity to observably increase performance by omitting unnecessary computations in the model, consider using the [filter pruning](./compression_algorithms/Pruning.md) algorithm. Models compressed with this algorithm can be executed more efficiently within OpenVINO Inference Engine runtime when compared to the uncompressed counterparts.

### Can I use NNCF-compressed models with runtimes other than OpenVINO Inference Engine?
While this is certainly possible in some cases, with a beneficial outcome even, we recommend NNCF as a way to get the most out of your setup based on OpenVINO Inference Engine inference.
We aim for best results on OpenVINO runtime with Intel hardware, and development-wise this is not always easy to generalize to other platforms or runtimes.

### Do I need OpenVINO or an Intel CPU to run NNCF?
Currently, this is not required in general.
Most NNCF backends can run compression and produce a compressed model object without OpenVINO or an Intel CPU on board of the machine. You only need OpenVINO and Intel hardware when you actually need to run inference on the compressed model, e.g. in a production scenario.


## PyTorch
### Importing anything from `nncf.torch` hangs
NNCF utilizes the [torch C++ extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html) mechanism to accelerate the quantization-aware training process.
This is done by just-in-time compiling a set of C++/CUDA files using the system-local compilers and toolsets.
The compilation happens at the first import of `nncf.torch` or anything under that namespace on the machine, or within the current Python environment.
The result is a set of the `.so` files containing the compiled extensions binary code stored in a system-specific location (commonly `~/.cache/torch_extensions`, or alternatively wherever `TORCH_EXTENSIONS_DIR` environment variable points to).

To avoid race conditions, PyTorch uses `.lock` files in this folder during compilation.
Sometimes, when the compilation process is abnormally aborted, these `.lock` files remain in the filesystem, which leads to a hang the next time you import anything from `nncf.torch`, because the just-in-time compilation process will wait indefinitely until the `.lock` files have been cleared.
To resolve these, delete the `torch_extensions` directory (at `~/.cache`, or pointed to by `TORCH_EXTENSIONS_DIR`, or at your specific location), and re-run the script that imports from `nncf.torch`.
The compilation takes some time and happens upon import, so do not interrupt the launch of your Python script until the import has been completed.

### Importing anything from `nncf.torch` leads to an error mentioning `gcc`, `nvcc`, or `ninja`
See the answer above for the general description of the reasons why these are involved in NNCF PyTorch operation.
To resolve, make sure that your CUDA installation contains the development tools (e.g. the `nvcc` compiler), and that the environmental variables are set properly so that these tools are available in `PATH` or `PYTHONPATH`.

### My model trains and runs slower in PyTorch when compressed by NNCF
NNCF does not in general accelerate training or inference when the compressed model is run in PyTorch.
It only prepares the model for further inference with OpenVINO's Inference Engine, where the runtime has capabilities of processing the NNCF-compressed models so that they run faster than their uncompressed counterparts.

The process of compressing in PyTorch relies on hooking regular PyTorch functions and calling extra code for purposes of compression algorithm logic, so the NNCF-processed models will inevitably run slower in PyTorch. Export your model after processing with NNCF to an OpenVINO-ingestable format (e.g. ONNX) and run it with the OpenVINO Inference Engine, to enjoy speedups when compared to the uncompressed model inference with Inference Engine.

### The .pth checkpoints for the compressed model have larger size and parameter count when compared to the uncompressed model
See the answer to the above question. Additional parameters are part of the compression algorithm internal state being saved along with the regular model weights, and any model size footprint reduction is deferred until exporting and/or running the model with OpenVINO Inference Engine.

### My RNN model is not compressed completely or fails at the compression stage
Currently NNCF PyTorch can only properly handle models with acyclic execution graphs.
RNNs, which inherently have cycles, can behave oddly when processed with NNCF PyTorch, which includes loss of quality, unreproducible results and failure to compress.

## TensorFlow

*To be filled*

## ONNX

*To be filled*

