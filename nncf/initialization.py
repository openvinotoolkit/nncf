import logging
from collections import OrderedDict
from functools import partial
from typing import Dict, Tuple, Any, Callable

import torch
from torch.nn.modules.loss import _Loss
from tqdm import tqdm

from nncf.nncf_logger import logger as nncf_logger
from nncf.quantization.init_range import MinMaxInitializer, ThreeSigmaInitializer, MeanMinMaxInitializer
from nncf.quantization.init_range import PercentileInitializer
from nncf.structures import QuantizationPrecisionInitArgs, QuantizationRangeInitArgs, BNAdaptationInitArgs
from nncf.utils import objwalk, is_tensor


class RangeInitializerFactory:
    @staticmethod
    def create(init_config: Dict, module: torch.nn.Module, log_module_name: str):
        init_type = init_config["type"]
        num_init_steps = init_config["num_init_steps"]
        if init_type == "min_max":
            return MinMaxInitializer(module, num_init_steps, log_module_name)
        if init_type == "threesigma":
            return ThreeSigmaInitializer(module, num_init_steps, log_module_name)
        if init_type == "mean_min_max":
            return MeanMinMaxInitializer(module, num_init_steps, log_module_name)
        if init_type == "percentile":
            min_percentile = init_config.get("min_percentile", 10)
            max_percentile = init_config.get("max_percentile", 90)
            return PercentileInitializer(module, num_init_steps, min_percentile, max_percentile, log_module_name)
        raise NotImplementedError


class InitializingDataLoader:
    """
    This class wraps the torch.utils.data.DataLoader class,
    and defines methods to parse the general data loader output to
    separate the input to the compressed model and the ground truth target
    for the neural network. This is required for proper initialization of
    certain compression algorithms.
    """

    def __init__(self, regular_data_loader):
        self.data_loader = regular_data_loader

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        return self

    def __next__(self) -> Any:
        loaded_item = next(self.data_loader_iter)
        return loaded_item

    def get_inputs(self, dataloader_output: Any) -> Tuple[Tuple, Dict]:
        """Returns (args, kwargs) for the current model call to be made during the initialization process"""
        raise NotImplementedError

    def get_target(self, dataloader_output: Any) -> Any:
        """Parses the generic data loader output and returns a structure to be used as
        ground truth in the loss criterion.
        :param dataloader_output - the (args, kwargs) tuple returned by the __next__ method."""

        raise NotImplementedError


class DefaultInitializingDataLoader(InitializingDataLoader):

    def get_inputs(self, dataloader_output: Any) -> Tuple[Tuple, Dict]:
        return (dataloader_output[0],), {}

    def get_target(self, dataloader_output: Any):
        return dataloader_output[1]


def wrap_dataloader_for_init(data_loader) -> InitializingDataLoader:
    if not isinstance(data_loader, InitializingDataLoader):
        loaded_item = next(iter(data_loader))
        if isinstance(loaded_item, (tuple, list)) and len(loaded_item) == 2:
            return DefaultInitializingDataLoader(data_loader)
        raise NotImplementedError("By default it is assumed that the data loader used for initialize "
                                  "produces a tuple/list of (*model_input*, *ground_truth*) and that no special "
                                  "forward arguments have to be set during init. If this is not the case, then instead "
                                  "of your regular data loader you need to pass a specialized version of "
                                  "InitializingDataLoader that returns a general (args, kwargs) tuple for your "
                                  "model to be called with at each __next__ call.")
    return data_loader


class DataLoaderBaseRunner:
    def __init__(self, model, init_device: str):
        self.model = model
        self.init_device = init_device
        self.progressbar_description = 'Algorithm initialization'

    def _run_model_inference(self, data_loader, num_init_steps, device):
        bar_format = '{l_bar}{bar} |{n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        for i, loaded_item in tqdm(
                enumerate(data_loader),
                total=num_init_steps,
                desc=self.progressbar_description,
                bar_format=bar_format,
        ):
            if num_init_steps is not None and i >= num_init_steps:
                break
            args_kwargs_tuple = data_loader.get_inputs(loaded_item)
            self._infer_batch(args_kwargs_tuple, device)

    def _infer_batch(self, args_kwargs_tuple, device):
        to_device_fn = partial(torch.Tensor.to, device=device)
        args, kwargs = objwalk(args_kwargs_tuple, is_tensor, to_device_fn)
        self.model(*args, **kwargs)

    def run(self, data_loader, num_init_steps):
        original_device = next(iter(self.model.parameters())).device
        self.model.to(self.init_device)

        class TQDMStream:
            @classmethod
            def write(cls, msg):
                tqdm.write(msg, end='')

        stream_handler = logging.StreamHandler(TQDMStream)
        nncf_logger.addHandler(stream_handler)

        self._prepare_initialization()
        device = next(self.model.parameters()).device
        data_loader = wrap_dataloader_for_init(data_loader)

        with torch.no_grad():
            self._run_model_inference(data_loader, num_init_steps, device)
            nncf_logger.removeHandler(stream_handler)
            self._apply_initializers()

        self.model.to(original_device)

    def _prepare_initialization(self):
        raise NotImplementedError

    def _apply_initializers(self):
        raise NotImplementedError


class DataLoaderRangeInitializeRunner(DataLoaderBaseRunner):
    def __init__(
            self,
            model,
            modules_to_init_vs_init_configs: Dict[str, Tuple[torch.nn.Module, Dict]],
            init_device: str,
    ):
        super().__init__(model, init_device)
        self.modules_to_init = modules_to_init_vs_init_configs
        self.progressbar_description = 'Range parameters initialization'
        self.initializers = OrderedDict()
        self.hook_handles = []

    def _prepare_initialization(self):
        for name, data in self.modules_to_init.items():
            module, init_config = data
            self.initializers[name] = RangeInitializerFactory.create(
                init_config, module, log_module_name=name
            )
            self.hook_handles.append(
                module.register_forward_hook(self.initializers[name].forward_hook)
            )

    def _apply_initializers(self):
        for handle in self.hook_handles:
            handle.remove()
        for initializer in self.initializers.values():
            initializer.apply_init()


class DataLoaderBNAdaptationRunner(DataLoaderBaseRunner):
    def __init__(self, model, init_device: str, num_bn_forget_steps):
        super().__init__(model, init_device)
        self.progressbar_description = 'BatchNorm statistics adaptation'
        self.num_bn_forget_steps = num_bn_forget_steps
        self.momentum_bn_forget = 0.9
        self.momentum_base = 0.1

    def _run_model_inference(self, data_loader, num_init_steps, device):
        num_bn_forget_steps = self.num_bn_forget_steps
        bar_format = '{l_bar}{bar} |{n_fmt}/{total_fmt} [{elapsed}<{remaining}]'

        def set_bn_momentum(module, momentum_value):
            if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
                module.momentum = momentum_value

        self.model.apply(partial(set_bn_momentum,
                                 momentum_value=self.momentum_bn_forget))

        for i, loaded_item in enumerate(data_loader):
            if num_bn_forget_steps is not None and i >= num_bn_forget_steps:
                break
            args_kwargs_tuple = data_loader.get_inputs(loaded_item)
            self._infer_batch(args_kwargs_tuple, device)

        self.model.apply(partial(set_bn_momentum,
                                 momentum_value=self.momentum_base))

        for i, loaded_item in tqdm(
                enumerate(data_loader),
                total=num_init_steps,
                desc=self.progressbar_description,
                bar_format=bar_format,
        ):
            if num_init_steps is not None and i >= num_init_steps:
                break
            args_kwargs_tuple = data_loader.get_inputs(loaded_item)
            self._infer_batch(args_kwargs_tuple, device)

    def _prepare_initialization(self):
        pass

    def _apply_initializers(self):
        pass


def default_criterion_fn(outputs: Any, target: Any, criterion: Any) -> torch.Tensor:
    return criterion(outputs, target)


def register_default_init_args(nncf_config: 'NNCFConfig',
                               train_loader,
                               criterion: _Loss = None,
                               criterion_fn: Callable[[Any, Any, _Loss], torch.Tensor] = None,
                               device='cuda') -> 'NNCFConfig':
    if criterion:
        if not criterion_fn:
            criterion_fn = default_criterion_fn
        nncf_config.register_extra_structs([QuantizationPrecisionInitArgs(criterion_fn=criterion_fn,
                                                                          criterion=criterion,
                                                                          data_loader=train_loader,
                                                                          device=device),
                                            QuantizationRangeInitArgs(data_loader=train_loader,
                                                                      device=device),
                                            BNAdaptationInitArgs(data_loader=train_loader,
                                                                 device=device)])
    else:
        nncf_config.register_extra_structs([QuantizationRangeInitArgs(data_loader=train_loader,
                                                                      device=device),
                                            BNAdaptationInitArgs(data_loader=train_loader,
                                                                 device=device)])
    return nncf_config
