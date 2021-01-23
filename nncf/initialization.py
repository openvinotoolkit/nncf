import math


from functools import partial
from typing import Dict, Tuple, Any, Callable, Optional

import torch
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss

from nncf.progress_bar import ProgressBar
from nncf.structures import QuantizationPrecisionInitArgs, QuantizationRangeInitArgs, \
    BNAdaptationInitArgs, AutoQPrecisionInitArgs
from nncf.utils import objwalk, is_tensor, training_mode_switcher


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
        self.batch_size = regular_data_loader.batch_size

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        return self

    def __next__(self) -> Any:
        loaded_item = next(self.data_loader_iter)
        return loaded_item

    def __len__(self):
        return len(self.data_loader)

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


class PartialDataLoader:
    def __init__(self, regular_data_loader: DataLoader, iter_ratio=1.0):
        if iter_ratio < 0.0 or iter_ratio > 1.0:
            raise ValueError("iter_ratio must be within 0 to 1 range")
        self.data_loader = regular_data_loader
        self.batch_size = regular_data_loader.batch_size
        self._stop_id = math.ceil(len(self.data_loader)*iter_ratio)
        self._batch_id = 0

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self._batch_id = 0
        return self

    def __next__(self) -> Any:
        if self._batch_id < self._stop_id:
            loaded_item = next(self.data_loader_iter)
            self._batch_id += 1
            return loaded_item
        raise StopIteration

    def __len__(self) -> int:
        return self._stop_id


class DataLoaderBaseRunner:
    def __init__(self, model, init_device: Optional[str]):
        self.model = model
        self.init_device = init_device
        self.progressbar_description = 'Algorithm initialization'

    def _run_model_inference(self, data_loader, num_init_steps, device):
        for i, loaded_item in ProgressBar(
                enumerate(data_loader),
                total=num_init_steps,
                desc=self.progressbar_description,
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
        if self.init_device is not None:
            original_device = next(iter(self.model.parameters())).device
            self.model.to(self.init_device)

        self._prepare_initialization()
        device = next(self.model.parameters()).device
        data_loader = wrap_dataloader_for_init(data_loader)

        with torch.no_grad():
            self._run_model_inference(data_loader, num_init_steps, device)
            self._apply_initializers()

        if self.init_device is not None:
            self.model.to(original_device)

    def _prepare_initialization(self):
        raise NotImplementedError

    def _apply_initializers(self):
        raise NotImplementedError


class SimpleDataLoaderRunner(DataLoaderBaseRunner):
    def _prepare_initialization(self):
        pass
    def _apply_initializers(self):
        pass


class DataLoaderBNAdaptationRunner(DataLoaderBaseRunner):
    def __init__(self, model, init_device: str, num_bn_forget_steps):
        super().__init__(model, init_device)
        self.progressbar_description = 'BatchNorm statistics adaptation'
        self.num_bn_forget_steps = num_bn_forget_steps
        self.momentum_bn_forget = 0.9
        self.original_momenta_values = {}

    @staticmethod
    def _apply_to_batchnorms(func):
        def func_apply_to_bns(module):
            if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
                func(module)
        return func_apply_to_bns

    def _run_model_inference(self, data_loader, num_init_steps, device):
        num_bn_forget_steps = self.num_bn_forget_steps

        def set_bn_momentum(module, momentum_value):
            module.momentum = momentum_value

        def save_original_bn_momenta(module):
            self.original_momenta_values[module] = module.momentum

        def restore_original_bn_momenta(module):
            module.momentum = self.original_momenta_values[module]

        with training_mode_switcher(self.model, is_training=True):
            self.model.apply(self._apply_to_batchnorms(save_original_bn_momenta))
            self.model.apply(self._apply_to_batchnorms(partial(set_bn_momentum,
                                                               momentum_value=self.momentum_bn_forget)))

            for i, loaded_item in enumerate(data_loader):
                if num_bn_forget_steps is not None and i >= num_bn_forget_steps:
                    break
                args_kwargs_tuple = data_loader.get_inputs(loaded_item)
                self._infer_batch(args_kwargs_tuple, device)

            self.model.apply(self._apply_to_batchnorms(restore_original_bn_momenta))

            for i, loaded_item in ProgressBar(
                    enumerate(data_loader),
                    total=num_init_steps,
                    desc=self.progressbar_description
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
                               train_loader: torch.utils.data.DataLoader,
                               criterion: _Loss = None,
                               criterion_fn: Callable[[Any, Any, _Loss], torch.Tensor] = None,
                               autoq_eval_fn: Callable[[torch.nn.Module, torch.utils.data.DataLoader], float] = None,
                               autoq_eval_loader: torch.utils.data.DataLoader = None,
                               device: str = None) -> 'NNCFConfig':

    nncf_config.register_extra_structs([QuantizationRangeInitArgs(data_loader=train_loader,
                                                                  device=device),
                                        BNAdaptationInitArgs(data_loader=train_loader,
                                                             device=device)])

    if criterion:
        if not criterion_fn:
            criterion_fn = default_criterion_fn
        nncf_config.register_extra_structs([QuantizationPrecisionInitArgs(criterion_fn=criterion_fn,
                                                                          criterion=criterion,
                                                                          data_loader=train_loader,
                                                                          device=device)])

    if autoq_eval_fn:
        if not autoq_eval_loader:
            autoq_eval_loader = train_loader
        nncf_config.register_extra_structs([AutoQPrecisionInitArgs(data_loader=autoq_eval_loader,
                                                                   eval_fn=autoq_eval_fn,
                                                                   nncf_config=nncf_config)])

    return nncf_config
