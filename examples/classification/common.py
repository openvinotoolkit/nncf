import torch.backends.cudnn as cudnn

from examples.common.sample_config import SampleConfig
from examples.common.execution import ExecutionMode, get_device
from examples.common.distributed import configure_distributed
from examples.common.model_loader import load_resuming_model_state_dict_and_checkpoint_from_path
from nncf.utils import manual_seed


def configure_device(current_gpu, config: SampleConfig):
    config.current_gpu = current_gpu
    config.distributed = config.execution_mode in (ExecutionMode.DISTRIBUTED, ExecutionMode.MULTIPROCESSING_DISTRIBUTED)
    if config.distributed:
        configure_distributed(config)

    config.device = get_device(config)


def set_seed(config):
    if config.seed is not None:
        manual_seed(config.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


def load_resuming_checkpoint(resuming_checkpoint_path):
    resuming_model_sd = None
    resuming_checkpoint = None
    if resuming_checkpoint_path is not None:
        resuming_model_sd, resuming_checkpoint = load_resuming_model_state_dict_and_checkpoint_from_path(
            resuming_checkpoint_path)
    return resuming_model_sd, resuming_checkpoint
