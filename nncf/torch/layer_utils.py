import torch
import torch.nn as nn

from nncf.common.utils.registry import Registry

COMPRESSION_MODULES = Registry('compression modules')


class ProxyModule:
    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)


class _NNCFModuleMixin:
    """
    Default class for modules that will be optimized by NNCF.

        Attributes:
            op_func_name    Name of corresponding torch function.
            target_weight_dim_for_compression   Target dimension of weights that will be compressed in some algorithms.
            ignored_algorithms   List of algorithms that will skip the module.
            _custom_forward_fn  wrapper of the custom forward function that is called with `self` argument equals to the
                ProxyModule
    """

    op_func_name = ""
    target_weight_dim_for_compression = 0
    _custom_forward_fn = None
    ignored_algorithms = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _NNCFModuleMixin.add_mixin_fields(self)

    @staticmethod
    def add_mixin_fields(obj):
        obj.pre_ops = nn.ModuleDict()
        obj.post_ops = nn.ModuleDict()

    def get_pre_op(self, key):
        return self.pre_ops[key]

    def get_post_op(self, key):
        return self.post_ops[key]

    def register_pre_forward_operation(self, op):
        key = str(len(self.pre_ops))
        self.pre_ops[key] = op
        return key

    def remove_pre_forward_operation(self, key):
        return self.pre_ops.pop(key)

    def register_post_forward_operation(self, op):
        key = str(len(self.post_ops))
        self.post_ops[key] = op
        return key

    def remove_post_forward_operation(self, key):
        return self.post_ops.pop(key)

    def reset(self):
        self.pre_ops.clear()
        self.post_ops.clear()

    def forward(self, *args):
        proxy_module = ProxyModule(self)
        for op in self.pre_ops.values():
            op_args = op(proxy_module, args)
            if op_args is not None:
                if not isinstance(op_args, tuple):
                    op_args = tuple([op_args])
                args = op_args
        forward_fn = self._custom_forward_fn.__func__ if self._custom_forward_fn else super().forward.__func__
        results = forward_fn(proxy_module, *args)
        for op in self.post_ops.values():
            op_results = op(proxy_module, results)
            if op_results is not None:
                results = op_results
        return results


class CompressionParameter(nn.Parameter):
    """
    The class that should be used in all compression algorithms instead of torch.nn.Parameter.

    This class utilize `compression_lr_multiplier` parameter from :class:`nncf.NNCFConfig`
    to increase/decrease gradients for compression algorithms' parameters.
    """

    def __new__(cls, data: torch.Tensor = None, requires_grad: bool = True, compression_lr_multiplier: float = None):
        return super().__new__(cls, data, requires_grad=requires_grad)

    def __init__(self, data: torch.Tensor = None, requires_grad: bool = True, compression_lr_multiplier: float = None):
        """

        Args:
            data: Parameter tensor
            requires_grad: If the parameter requires gradient
            compression_lr_multiplier: Multiplier for gradient values
        """
        super().__init__()

        if compression_lr_multiplier is not None and self.dtype.is_floating_point:
            self.requires_grad = True
            self.register_hook(lambda grad: compression_lr_multiplier * grad)
            self.requires_grad = requires_grad
