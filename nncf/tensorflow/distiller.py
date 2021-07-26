
from typing import Dict, Any

import tensorflow as tf

from nncf.common.statistics import NNCFStatistics
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from nncf.api.compression import CompressionLoss, CompressionAlgorithmController


class Distiller(tf.keras.Model):
    def __init__(self, compressed_model, original_model):
        super().__init__()
        self.compressed_model = compressed_model
        self.original_model = original_model
        self.kd_storage = None
        self.kd_loss_fn = tf.keras.losses.MeanSquaredError()

    def call(self, inputs, training=None, mask=None):
        compressed_output = self.compressed_model(inputs)
        kd_outputs = self.original_model(inputs)
        self.kd_storage = self.kd_loss_fn(compressed_output, kd_outputs)
        return compressed_output
        # distilaltion loss just calls kd_storage

    def get_knowledge_distillation_loss_value(self):
        return self.kd_storage


class KnowledgeDistillationLoss(CompressionLoss):
    def __init__(self, target_model, original_model):
        self.target_model = target_model
        self.target_model.enable_knowledge_distillation(original_model)

    def calculate(self, *args, **kwargs):
        return self.target_model.get_knowledge_distillation_loss_value()

    def load_state(self, state: Dict[str, Any]) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}


@TF_COMPRESSION_ALGORITHMS.register('knowledge_distillation')
class KnowledgeDistillationBuilder(TFCompressionAlgorithmBuilder):

    def get_transformation_layout(self, model) -> TransformationLayout:
        self.original_model = tf.keras.models.clone_model(model)
        return TransformationLayout()

    def _build_controller(self, model: tf.keras.Model) -> CompressionAlgorithmController:
        return KnowledgeDistilaltionController(model, self.original_model)

    def initialize(self, model: tf.keras.Model) -> None:
        pass

    def _get_algo_specific_config_section(self) -> Dict:
        return {}


class KnowledgeDistilaltionController(BaseCompressionAlgorithmController):
    def __init__(self, target_model: tf.keras.Model, original_model: tf.keras.Model):
        super().__init__(target_model)
        self._loss = KnowledgeDistillationLoss(target_model, original_model)
        self._scheduler = StubCompressionScheduler()

    @property
    def loss(self) -> KnowledgeDistillationLoss:
        return self._loss

    @property
    def scheduler(self) -> StubCompressionScheduler:
        return self._scheduler

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()
