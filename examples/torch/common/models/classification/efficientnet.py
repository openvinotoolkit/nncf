# Efficient Net implementation from: https://github.com/lukemelas/EfficientNet-PyTorch

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding

from nncf.torch import register_module

wrapper = register_module()
wrapper(Conv2dStaticSamePadding)


def efficient_net(pretrained=True, num_classes=1000, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained(num_classes=num_classes, **kwargs)
    return EfficientNet.from_name(num_classes=num_classes, **kwargs)
