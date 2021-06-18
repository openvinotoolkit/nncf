from typing import Type

from nncf.torch.quantization.precision_init.autoq_init import AutoQPrecisionInitializer
from nncf.torch.quantization.precision_init.base_init import BasePrecisionInitializer
from nncf.torch.quantization.precision_init.hawq_init import HAWQPrecisionInitializer
from nncf.torch.quantization.precision_init.manual_init import ManualPrecisionInitializer


class PrecisionInitializerFactory:
    @staticmethod
    def create(init_type: str) -> Type[BasePrecisionInitializer]:
        if init_type == "manual":
            return ManualPrecisionInitializer
        if init_type == "hawq":
            return HAWQPrecisionInitializer
        if init_type == "autoq":
            return AutoQPrecisionInitializer
        raise NotImplementedError
