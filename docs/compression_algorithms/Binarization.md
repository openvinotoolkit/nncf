>_Scroll down for the examples of the JSON configuration files that can be used to apply this algorithm_.
### Binarization
NNCF supports binarizing weights and activations for 2D convolutional PyTorch\* layers (Conv2D) *only*.

Weight binarization may be done in two ways, depending on the configuration file parameters - either via [XNOR binarization](https://arxiv.org/abs/1603.05279) or via [DoReFa binarization](https://arxiv.org/abs/1606.06160). For DoReFa binarization, the scale of binarized weights for each convolution operation is calculated as the mean of absolute values of non-binarized convolutional filter weights, while for XNOR binarization, each convolutional operation has scales that are calculated in the same manner, but _per input channel_ of the convolutional filter. Refer to the original papers for details.

Binarization of activations is implemented via binarizing inputs to the convolutional layers in the following way:

$\text{out} = s * H(\text{in} - s*t)$

In the formula above,
 - $\text{in}$ - non-binarized activation values
 - $\text{out}$ - binarized activation values
 - $H(x)$ is the Heaviside step function
 - $s$ and $t$ are trainable parameters corresponding to binarization scale and threshold respectively

Training binarized networks requires special scheduling of the training process. For instance, binarizing a pretrained ResNet18 model on ImageNet is a four-stage process, with each stage taking a certain number of epochs. During the stage 1, the network is trained without any binarization. During the stage 2, the training continues with binarization enabled for activations only. During the stage 3, binarization is enabled both for activations and weights. Finally, during the stage 4 the optimizer learning rate, which was kept constant at previous stages, is decreased according to a polynomial law, while weight decay parameter of the optimizer is set to 0. The configuration files for the NNCF binarization algorithm allow to control certain parameters of this training schedule.


### Example configuration files:

>_For the full list of the algorithm configuration parameters via config file, see the corresponding section in the [NNCF config schema](https://openvinotoolkit.github.io/nncf/)_.

- Binarize a ResNet using XNOR algorithm, ignoring several portions of the model, with finetuning on the scope of 60 epochs and staged binarization schedule (activations first, then weights)
```json5
{
    "input_info": { "sample_size": [1, 3, 224, 224] },
    "compression":
    {
        "algorithm": "binarization",
        "mode": "xnor",
        "params": {
            "activations_quant_start_epoch": 10,  // Epoch to start binarizing activations
            "weights_quant_start_epoch": 30,  // Epoch to start binarizing weights
            "lr_poly_drop_start_epoch": 60,  // Epoch to start dropping the learning rate
            "lr_poly_drop_duration_epochs": 30,  // Duration, in epochs, of the learning rate dropping process.
            "disable_wd_start_epoch": 60  // Epoch to disable weight decay in the optimizer
        },

        "ignored_scopes": ["ResNet/NNCFLinear[fc]/linear_0",
                           "ResNet/NNCFConv2d[conv1]/conv2d_0",
                           "ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/NNCFConv2d[0]/conv2d_0",
                           "ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/NNCFConv2d[0]/conv2d_0",
                           "ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/NNCFConv2d[0]/conv2d_0"]
    }
}
```
