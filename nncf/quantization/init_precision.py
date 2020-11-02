from .precision_init.hawq_init import HAWQPrecisionInitializer
from .precision_init.manual_init import ManualPrecisionInitializer


class PrecisionInitializerFactory:
    @staticmethod
    def create(init_type: str):
        if init_type == "manual":
            return ManualPrecisionInitializer
        if init_type == "hawq":
            return HAWQPrecisionInitializer
        raise NotImplementedError
