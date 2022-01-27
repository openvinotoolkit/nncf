from enum import Enum


class ElasticityDim(Enum):
    KERNEL = 'kernel'
    WIDTH = 'width'
    DEPTH = 'depth'

    @classmethod
    def from_str(cls, dim: str) -> 'ElasticityDim':
        if dim == ElasticityDim.KERNEL.value:
            return ElasticityDim.KERNEL
        if dim == ElasticityDim.WIDTH.value:
            return ElasticityDim.WIDTH
        if dim == ElasticityDim.DEPTH.value:
            return ElasticityDim.DEPTH
        raise RuntimeError(f"Unknown elasticity dimension: {dim}."
                           f"List of supported: {[e.value for e in ElasticityDim]}")
