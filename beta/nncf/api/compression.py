"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from nncf.tensorflow.graph.model_transformer import ModelTransformer
from nncf.configs.config import Config
from nncf.tensorflow.utils.save import save_model


class CompressionLoss:
    """
    Used to calculate additional loss to be added to the base loss during the
    training process. It uses the model graph to measure variables and activations
    values of the layers during the loss construction. For example, the $L_0$-based
    sparsity algorithm calculates the number of non-zero weights in convolutional
    and fully-connected layers to construct the loss function.
    """

    def call(self):
        """
        Returns the compression loss value.
        """
        return 0

    def statistics(self):
        """
        Returns a dictionary of printable statistics.
        """
        return {}

    def __call__(self, *args, **kwargs):
        """
        Invokes the `CompressionLoss` instance.
        Returns:
            the compression loss value.
        """
        return self.call(*args, **kwargs)

    def get_config(self):
        """
        Returns the config dictionary for a `CompressionLoss` instance.
        """
        return {}

    @classmethod
    def from_config(cls, config):
        """
        Instantiates a `CompressionLoss` from its config (output of `get_config()`).
        Arguments:
            config: Output of `get_config()`.
        Returns:
            A `CompressionLoss` instance.
        """
        return cls(**config)


class CompressionScheduler:
    """
    Implements the logic of compression method control during the training process.
    May change the method hyperparameters in regards to the current training step or
    epoch. For example, the sparsity method can smoothly increase the sparsity rate
    over several epochs.
    """

    def __init__(self):
        self.last_epoch = -1
        self.last_step = -1

    def step(self, last=None):
        """
        Should be called at the beginning of each training step.
        Arguments:
            `last` - specifies the initial "previous" step.
        """
        if last is None:
            last = self.last_step + 1
        self.last_step = last

    def epoch_step(self, last=None):
        """
        Should be called at the beginning of each training epoch.
        Arguments:
            `last` - specifies the initial "previous" epoch.
        """
        if last is None:
            last = self.last_epoch + 1
        self.last_epoch = last

    def load_state(self, initial_step, steps_per_epoch):
        self.last_step = initial_step - 1
        self.last_epoch = self.last_step // steps_per_epoch

    def get_config(self):
        """
        Returns the config dictionary for a `CompressionScheduler` instance.
        """
        return {}

    @classmethod
    def from_config(cls, config):
        """
        Instantiates a `CompressionScheduler` from its config (output of `get_config()`).
        Arguments:
            config: Output of `get_config()`.
        Returns:
            A `CompressionScheduler` instance.
        """
        return cls(**config)


class CompressionAlgorithmInitializer:
    """
    Configures certain parameters of the algorithm that require access to the dataset
    (for example, in order to do range initialization for activation quantizers) or
    to the loss function to be used during fine-tuning (for example, to determine
    quantizer precision bitwidth using HAWQ).
    """

    def call(self, model, dataset=None, loss=None):
        pass

    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)


class CompressionAlgorithmController:
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model in order to enable algorithm-specific compression.
    Hosts entities that are to be used during the training process, such as compression scheduler and
    compression loss.
    """

    def __init__(self, target_model):
        """
        Arguments:
          `target_model` - model with additional modifications necessary to enable algorithm-specific
                           compression during fine-tuning built by the `CompressionAlgorithmBuilder`.
        """
        self._model = target_model
        self._loss = CompressionLoss()
        self._scheduler = CompressionScheduler()
        self._initializer = CompressionAlgorithmInitializer()

    @property
    def model(self):
        return self._model

    @property
    def loss(self):
        return self._loss

    @property
    def scheduler(self):
        return self._scheduler

    def initialize(self, dataset=None, loss=None):
        """
        Configures certain parameters of the algorithm that require access to the dataset
        (for example, in order to do range initialization for activation quantizers) or
        to the loss function to be used during fine-tuning (for example, to determine
        quantizer precision bitwidth using HAWQ).
        """
        self._initializer(self._model, dataset, loss)

    def statistics(self):
        """
        Returns a dictionary of printable statistics.
        """
        return self._loss.statistics()

    def export_model(self, save_path, save_format='frozen_graph'):
        """
        Used to export the compressed model to the Frozen Graph, TensorFlow SavedModel or
        Keras H5 formats. Makes method-specific preparations of the model, (e.g. removing
        auxiliary layers that were used for the model compression), then exports the model
        in specified path.
        Arguments:
           `save_path` - a path to export model.
           `save_format` - saving format (`frozen_graph` for Frozen Graph,
           `tf` for Tensorflow SavedModel, `h5` for Keras H5 format).
        """
        stripped_model = self.strip_model(self.model)
        save_model(stripped_model, save_path, save_format)

    def strip_model(self, model):
        """
        Strips auxiliary layers that were used for the model compression, as it's only needed
        for training. The method is used before exporting model in the target format.
        Arguments:
            model: compressed model.
        Returns:
             A stripped model.
        """
        return model


class CompressionAlgorithmBuilder:
    """
    Determines which modifications should be made to the original model in
    order to enable algorithm-specific compression during fine-tuning.
    """

    def __init__(self, config: Config):
        """
        Arguments:
          `config` - a dictionary that contains parameters of compression method.
        """
        self.config = config

    def apply_to(self, model):
        """
        Applies algorithm-specific modifications to the model.
        Arguments:
            model: original uncompressed model.
        Returns:
             Model with additional modifications necessary to enable algorithm-specific
             compression during fine-tuning.
        """
        transformation_layout = self.get_transformation_layout(model)
        return ModelTransformer(model, transformation_layout).transform()

    def build_controller(self, model):
        """
        Builds `CompressionAlgorithmController` to handle to the additional modules, parameters
        and hooks inserted into the model in order to enable algorithm-specific compression.
        Arguments:
            model: model with additional modifications necessary to enable
                   algorithm-specific compression during fine-tuning.
        Returns:
            An instance of the `CompressionAlgorithmController`.
        """
        return CompressionAlgorithmController(model)

    def get_transformation_layout(self, model):
        """
        Computes necessary model transformations to enable algorithm-specific compression.
        Arguments:
            model: original uncompressed model.
        Returns:
            An instance of the `TransformationLayout` class containing a list of
            algorithm-specific modifications.
        """
        raise NotImplementedError
