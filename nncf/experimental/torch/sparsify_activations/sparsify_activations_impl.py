from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, TypeVar

import nncf
from nncf.common import factory
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.logging.track_progress import track
from nncf.common.scopes import matches_any
from nncf.common.scopes import should_consider_scope
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.data import Dataset
from nncf.scopes import IgnoredScope
from nncf.scopes import get_ignored_node_names_from_ignored_scope
from nncf.torch.model_creation import is_wrapped_model
from nncf.torch.model_creation import wrap_model

TModel = TypeVar("TModel")


class SparsifyActivationsAlgoBackend(ABC):

    def do_inference(self, model: TModel, dataset: Dataset):
        engine = factory.EngineFactory.create(model)
        for input_data in track(
            dataset.get_inference_data(),
            total=dataset.get_length(),
            description="Activation sparsifier calibration",
        ):
            engine.infer(input_data)

    @property
    @abstractmethod
    def supported_metatypes(self) -> List[OperatorMetatype]:
        pass

    @abstractmethod
    def insert_sparsifiers(self, model: TModel, target_sparsity_by_node: Dict[NNCFNode, float]) -> TModel:
        pass

    @abstractmethod
    def calibrate_sparsifiers(self, model: TModel, dataset: Dataset) -> TModel:
        pass

    @abstractmethod
    def freeze_sparsifiers(self, model: TModel) -> TModel:
        pass


class SparsifyActivationsAlgorithm:

    def __init__(
        self,
        target_sparsity_by_scope: dict[str, float],
        ignored_scope: IgnoredScope,
    ):
        self._target_sparsity_by_scope = target_sparsity_by_scope
        self._ignored_scope = ignored_scope
        self._backend_entity = None

    @property
    def available_backends(self) -> List[BackendType]:
        return [BackendType.TORCH]

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.TORCH:
            from nncf.experimental.torch.sparsify_activations.torch_backend import PTSparsifyActivationsAlgoBackend
            self._backend_entity = PTSparsifyActivationsAlgoBackend()
        else:
            raise nncf.UnsupportedBackendError(
                f"{model_backend.value} backend is not supported for `sparsify_activations`."
            )

    def _get_target_sparsity_by_node(self, nncf_graph: NNCFGraph) -> Dict[NNCFNode, float]:
        """
        Collects nodes in the model's graph corresponding to the layers for weight compression.

        :param nncf_graph: NNCFGraph instance.
        :return: List with the data for each layer.
        """
        supported_metatypes = self._backend_entity.supported_metatypes
        ignored_names = get_ignored_node_names_from_ignored_scope(
            self._ignored_scope, nncf_graph, strict=self._ignored_scope.validate
        )
        print(ignored_names)
        target_sparsity_by_node = {}
        for node in nncf_graph.topological_sort():
            print(node.metatype, node.node_name, ignored_names,
                  'should_consider_scope=',
                  should_consider_scope(node.node_name, ignored_names))
            if node.metatype not in supported_metatypes or not should_consider_scope(node.node_name, ignored_names):
                continue
            for scope, target_sparsity in self._target_sparsity_by_scope.items():
                if matches_any(node.node_name, scope):
                    if node.node_name in target_sparsity_by_node:
                        raise nncf.ValidationError(
                            f'"{node.node_name}" is matched by multiple items in `target_sparsity_by_scope`.')
                    target_sparsity_by_node[node] = target_sparsity
        return target_sparsity_by_node

    def do_sparsification(
        self,
        model: TModel,
        graph: NNCFGraph,
        target_sparsity_by_node: Dict[NNCFNode, float],
        dataset: Dataset,
    ):
        model = self._backend_entity.insert_sparsifiers(
            model, graph, target_sparsity_by_node)
        model = self._backend_entity.calibrate_sparsifiers(model, dataset)
        model = self._backend_entity.freeze_sparsifiers(model)
        return model

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        dataset: Dataset,
    ) -> TModel:
        self._set_backend_entity(model)
        target_sparsity_by_node = self._get_target_sparsity_by_node(graph)
        sparsified_model = self.do_sparsification(
            model, graph, target_sparsity_by_node, dataset,
        )
        return sparsified_model


def sparsify_activations(
    model: TModel,
    dataset: Dataset,
    target_sparsity_by_scope: Dict[str, float],
    ignored_scope: Optional[IgnoredScope] = None,
    debug_folder=None,
) -> TModel:
    """
    Implementation of the `compress_weights()` method.
    """

    backend = get_backend(model)
    if backend == BackendType.TORCH and not is_wrapped_model(model):
        example_input = next(iter(dataset.get_inference_data()))
        model = wrap_model(
            model,
            example_input=example_input,
            trace_parameters=True,
        )

    if ignored_scope is None:
        ignored_scope = IgnoredScope()

    algorithm = SparsifyActivationsAlgorithm(
        target_sparsity_by_scope, ignored_scope)

    graph = NNCFGraphFactory.create(model)
    if debug_folder:
        graph.dump_graph(
            Path(debug_folder, './before-sparsification.dot').as_posix())
    sparse_model = algorithm.apply(model, graph, dataset)
    graph = NNCFGraphFactory.create(sparse_model)
    if debug_folder:
        graph.dump_graph(
            Path(debug_folder, './after-sparsification.dot').as_posix())
    return sparse_model
