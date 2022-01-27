from abc import ABC
from abc import abstractmethod

from typing import TypeVar

from nncf.common.graph.graph import NNCFGraph

ModelType = TypeVar('ModelType')


class CompressedModel(ABC):
    def __init__(self, model: ModelType):
        self.original_model = model
        self.nncf_graph = self.build_nncf_graph()

    @abstractmethod
    def build_nncf_graph(self) -> NNCFGraph:
        ...
