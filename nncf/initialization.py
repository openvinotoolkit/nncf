import logging
from collections import OrderedDict
from typing import Dict, Tuple

import torch
from functools import partial

from nncf.structures import QuantizationPrecisionInitArgs, QuantizationRangeInitArgs
from tqdm import tqdm

from nncf.nncf_logger import logger as nncf_logger
from nncf.quantization.init_range import MinMaxInitializer, ThreeSigmaInitializer, MeanMinMaxInitializer
from nncf.quantization.init_range import PercentileInitializer
from nncf.utils import objwalk


def wrap_data_loader(data_loader, default_wrapper_cls, kwargs=None, device=None):
    # pylint: disable=unidiomatic-typecheck
    if type(data_loader) == torch.utils.data.DataLoader:
        wrapped_data_loader = default_wrapper_cls(data_loader=data_loader,
                                                  kwargs=kwargs,
                                                  device=device)
    else:
        wrapped_data_loader = data_loader
    return wrapped_data_loader


class RangeInitializerFactory:
    @staticmethod
    def create(init_config: Dict, module: torch.nn.Module, log_module_name: str):
        init_type = init_config["type"]
        if init_type == "min_max":
            return MinMaxInitializer(module, log_module_name)
        if init_type == "threesigma":
            return ThreeSigmaInitializer(module, log_module_name)
        if init_type == "mean_min_max":
            return MeanMinMaxInitializer(module, log_module_name)
        if init_type == "percentile":
            min_percentile = init_config["min_percentile"]
            max_percentile = init_config["max_percentile"]
            return PercentileInitializer(module, min_percentile, max_percentile, log_module_name)
        raise NotImplementedError


class InitializingDataLoader:
    """
    This class wraps the torch.utils.data.DataLoader class,
    enabling to return a tuple containing the input tensor
    and additional kwargs required for the forward pass
    when used as an iterator.
    Custom logic to prepare the input tensor might be
    added to the __next__ method.
    """

    def __init__(self, data_loader, kwargs, device):
        self.data_loader = data_loader
        self.kwargs = kwargs
        self.device = device

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        return self

    def __next__(self):
        batch_input, batch_target = next(self.data_loader_iter)
        tup = (batch_input, batch_target)

        def is_tensor(obj):
            return isinstance(obj, torch.Tensor)

        to_device_fn = partial(torch.Tensor.to, device=self.device)
        batch_input, batch_target = objwalk(tup, is_tensor, to_device_fn)
        return batch_input, batch_target, self.kwargs

    @property
    def num_workers(self):
        return self.data_loader.num_workers

    @num_workers.setter
    def num_workers(self, num_workers):
        self.data_loader.num_workers = num_workers


class DataLoaderInitializeRunner:
    def __init__(self, model, modules_to_init_vs_init_configs: Dict[str, Tuple[torch.nn.Module, Dict]]):
        super().__init__()
        self.model = model
        self.modules_to_init = modules_to_init_vs_init_configs

    def run(self, data_loader, num_init_steps, *args, **kwargs):
        class TQDMStream:
            @classmethod
            def write(cls, msg):
                tqdm.write(msg, end='')

        stream_handler = logging.StreamHandler(TQDMStream)
        nncf_logger.addHandler(stream_handler)
        device = next(self.model.parameters()).device
        wrapped_data_loader = wrap_data_loader(data_loader, InitializingDataLoader, kwargs, device)

        initializers = OrderedDict()
        hook_handles = []
        for name, data in self.modules_to_init.items():
            module, init_config = data
            initializers[name] = RangeInitializerFactory.create(init_config, module, log_module_name=name)
            hook_handles.append(module.register_forward_hook(initializers[name].forward_hook))
        with torch.no_grad():
            bar_format = '{l_bar}{bar} |{n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            bar_desc = 'Algorithm initialization'
            for i, (input_, _, dataloader_kwargs) in tqdm(enumerate(wrapped_data_loader), total=num_init_steps,
                                                          desc=bar_desc, bar_format=bar_format):
                if num_init_steps is not None and i >= num_init_steps:
                    break
                self.model(input_, **dataloader_kwargs)
            nncf_logger.removeHandler(stream_handler)
            for handle in hook_handles:
                handle.remove()
            for initializer in initializers.values():
                initializer.apply_init()

def register_default_init_args(nncf_config: 'NNCFConfig', criterion, train_loader) -> 'NNCFConfig':
    nncf_config.register_extra_structs([QuantizationPrecisionInitArgs(criterion=criterion, data_loader=train_loader),
                                        QuantizationRangeInitArgs(data_loader=train_loader)])
    return nncf_config
