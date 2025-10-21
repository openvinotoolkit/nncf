from typing import Optional

from nncf.common.quantization.backend_parameters import is_weight_compression_needed



class BackendParameters:
    COMPRESS_WEIGHTS = "compress_weights"
    STAT_REQUESTS_NUMBER = "stat_requests_number"
    EVAL_REQUESTS_NUMBER = "eval_requests_number"
    ACTIVATIONS = "activations"
    WEIGHTS = "weights"
    LEVEL_LOW = "level_low"
    LEVEL_HIGH = "level_high"


def is_weight_compression_needed(advanced_parameters: Optional[AdvancedQuantizationParameters]) -> bool:
    """
    Determines whether weight compression is needed based on the provided
    advanced quantization parameters.

    :param advanced_parameters: Advanced quantization parameters.
    :return: True if weight compression is needed, False otherwise.
    """
    if advanced_parameters is not None and advanced_parameters.backend_params is not None:
        return advanced_parameters.backend_params.get(BackendParameters.COMPRESS_WEIGHTS, True)
    return True
