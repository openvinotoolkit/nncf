import pytest
import torch
from torch import nn

from nncf.experimental.torch.replace_custom_modules.replace_custom_modules import (
    replace_custom_modules_with_torch_native,
)

try:
    import timm
    from timm.models.layers import Linear
    from timm.models.layers.norm_act import BatchNormAct2d
    from timm.models.layers.norm_act import GroupNormAct
    from timm.models.layers.norm_act import LayerNormAct

except ImportError:
    timm = None

if timm is not None:

    def _count_custom_modules(model) -> int:
        """
        Get number of custom modules in the model.

        :param model: The target model.
        :return int: Number of custom modules.
        """
        custom_types = [
            Linear,
            BatchNormAct2d,
            GroupNormAct,
            LayerNormAct,
        ]
        return len([m for _, m in model.named_modules() if type(m) in custom_types])

    TEST_CUSTOM_MODULES = [
        Linear(
            in_features=2,
            out_features=2,
        ),
        BatchNormAct2d(
            num_features=2,
            act_layer=nn.ReLU,
        ),
        GroupNormAct(
            num_channels=2,
            num_groups=2,
            act_layer=nn.ReLU,
        ),
        LayerNormAct(
            normalization_shape=(2, 2),
            act_layer=nn.ReLU,
        ),
    ]

    @pytest.mark.parametrize(
        "custom_module", TEST_CUSTOM_MODULES, ids=[m.__class__.__name__ for m in TEST_CUSTOM_MODULES]
    )
    @pytest.mark.skipif(timm is None, reason="Not install timm package")
    def test_replace_custom_timm_module(custom_module):
        """
        Test output of replaced module with custom module
        """
        native_module = replace_custom_modules_with_torch_native(custom_module)
        input_data = torch.rand(1, 2, 2, 2)
        out_custom = custom_module(input_data)
        out_native = native_module(input_data)

        assert type(custom_module) != type(native_module)
        assert torch.equal(out_custom, out_native)

    def test_replace_custom_modules_in_timm_model():
        """
        Test that all modules from timm model replaced by replace_custom_modules_with_torch_native
        """
        timm_model = timm.create_model(
            "mobilenetv3_small_050", num_classes=1000, in_chans=3, pretrained=True, checkpoint_path=""
        )
        input_data = torch.rand(1, 3, 224, 224)
        out_timm = timm_model(input_data)

        native_model = replace_custom_modules_with_torch_native(timm_model)
        out_native = native_model(input_data)
        assert torch.equal(out_timm, out_native)

        num_custom_modules_in_timm = _count_custom_modules(timm_model)
        num_custom_modules_in_native = _count_custom_modules(native_model)

        assert num_custom_modules_in_native == 0
        assert num_custom_modules_in_timm > 0
