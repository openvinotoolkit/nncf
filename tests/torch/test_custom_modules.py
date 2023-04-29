import torch
import torch.nn.functional

from nncf import NNCFConfig
from nncf.torch import register_module
from tests.torch.helpers import create_compressed_model_and_algo_for_test


@register_module()
class CustomConvModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones([1, 1, 1, 1]))

    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.weight)


class ModelWithCustomConvModules(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.regular_conv = torch.nn.Conv2d(1, 1, 1)
        self.custom_conv = CustomConvModule()

    def forward(self, x):
        x = self.regular_conv(x)
        x = self.custom_conv(x)
        return x


def test_custom_module_processing():
    nncf_config = NNCFConfig.from_dict({"input_info": {"sample_size": [1, 1, 1, 1]}})

    # Should complete successfully without exceptions:
    create_compressed_model_and_algo_for_test(ModelWithCustomConvModules(), nncf_config)
