from examples.torch.common.model_loader import load_resuming_model_state_dict_and_checkpoint_from_path


def load_resuming_checkpoint(resuming_checkpoint_path):
    resuming_model_sd = None
    resuming_checkpoint = None
    if resuming_checkpoint_path is not None:
        resuming_model_sd, resuming_checkpoint = load_resuming_model_state_dict_and_checkpoint_from_path(
            resuming_checkpoint_path)
    return resuming_model_sd, resuming_checkpoint
