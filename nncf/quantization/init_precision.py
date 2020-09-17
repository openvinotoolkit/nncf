from .precision_init.hawq_init import HAWQPrecisionInitializer
from .precision_init.manual_init import ManualPrecisionInitializer
from .precision_init.autoq_init import AutoQPrecisionInitializer

class PrecisionInitializerFactory:
    @staticmethod
    def create(init_type: str):
        if init_type == "manual":
            return ManualPrecisionInitializer
        if init_type == "hawq":
            return HAWQPrecisionInitializer
        if init_type == "autoq":
            return AutoQPrecisionInitializer
        raise NotImplementedError
