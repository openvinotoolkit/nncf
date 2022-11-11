# FracBits mixed-precision quantization algorithm

This provides sample configurations of FracBits mixed-precision quantization algorithm for image classification tasks.

## Prerequiste

Please follow [installation guide](../../../torch/classification/README.md#installation) and [dataset preperation guide](../../../torch/classification/README.md#dataset-preparation) of NNCF PyTorch classification examples.

## Compress FP32 model with FracBits

You can run the FracBits mixed-precision quantization algorithm with the pre-defined configuration file.

```bash
cd examples/experimental/torch/classification
python fracbits.py -m train -c <config_path> -j <num_workers> --data <dataset_path> --log-dir <path_for_logging>
```

The following describes each argument.

- `-c`: FracBits configuration file path. You can find it from `examples/experimental/torch/classification/fracbits_configs`.
- `-j`: The number of PyTorch dataloader workers.
- `--data`: Directory path of the dataset.
- `--log-dir`: Directory path to save log files, tensorboard logs, and model checkpoints.

We provide configurations for three model architectures: `inception_v3`, `mobilenet_v2`, and `resnet50`. Our configurations almost uses the ImageNet dataset except `mobilenet_v2` which also has a configuration for the CIFAR100 dataset.

## Results for FracBits

|    Model     | Compression algorithm | Dataset  | Accuracy (Drop) % |                                                                       NNCF config file                                                                        | Compression rate |
| :----------: | :-------------------: | :------: | :---------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------: |
| MobileNet-V2 |       FracBits        | CIFAR100 |   67.26 (0.45)    | [mobilenet_v2_cifar100_mixed_int_fracbits_msize.json](./configs/mobilenet_v2_cifar100_mixed_int_fracbits_msize.json) |       1.5        |
| Inception-V3 |       FracBits        | ImageNet |   78.16 (-0.82)   | [inception_v3_imagenet_mixed_int_fracbits_msize.json](./configs/inception_v3_imagenet_mixed_int_fracbits_msize.json) |       1.51       |
| MobileNet-V2 |       FracBits        | ImageNet |   71.19 (0.68)    | [mobilenet_v2_imagenet_mixed_int_fracbits_msize.json](./configs/mobilenet_v2_imagenet_mixed_int_fracbits_msize.json) |       1.53       |
|  ResNet-50   |       FracBits        | ImageNet |   76.12 (0.04)    |     [resnet50_imagenet_mixed_int_fracbits_msize.json](./configs/resnet50_imagenet_mixed_int_fracbits_msize.json)     |       1.54       |

- We used a NVIDIA V100 x 8 machine to obtain all results except MobileNet-V2, CIFAR100 experiment.
- Model accuracy is obtained by averaging on 5 repeats.
- Absolute accuracy drop is compared to FP32 model accuracy reported in [Results for quantization](../../../torch/classification/README.md#results-for-quantization).
- Compression rate is about the reduced model size compared to the initial one. The model initial state starts from INT8 quantization, so compression rate = 1.5 means that the model size is reduced to 2/3 compared to the INT8 model.
- Model size is the total number of bits in model weights. It is computed by $\sum_i \textrm{num-params}_i \times \textrm{bitwidth}_i$ where $\textrm{num-params}_i$ is the number of parameters of $i$-th layer and $\textrm{bitwidth}_i$ is the bit-width of $i$-th layer.
