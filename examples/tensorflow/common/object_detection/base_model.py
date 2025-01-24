# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import re

import tensorflow as tf

from examples.tensorflow.common.object_detection import checkpoint_utils


def _make_filter_trainable_variables_fn(frozen_variable_prefix):
    """Creates a function for filtering trainable varialbes."""

    def _filter_trainable_variables(variables):
        """Filters trainable varialbes.

        Args:
            variables: a list of tf.Variable to be filtered.

        Returns:
            filtered_variables: a list of tf.Variable filtered out the frozen ones.
        """
        # frozen_variable_prefix: a regex string specifing the prefix pattern of
        # the frozen variables' names.
        filtered_variables = [
            v for v in variables if not frozen_variable_prefix or not re.match(frozen_variable_prefix, v.name)
        ]
        return filtered_variables

    return _filter_trainable_variables


class Model:
    """Base class for model function."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, params):
        # One can use 'RESNET_FROZEN_VAR_PREFIX' to speed up ResNet training when loading from the checkpoint
        # RESNET_FROZEN_VAR_PREFIX = r'(resnet\d+)\/(conv2d(|_([1-9]|10))|batch_normalization(|_([1-9]|10)))\/'
        self._frozen_variable_prefix = ""
        params_train_regularization_variable_regex = r".*(kernel|weight):0$"
        self._regularization_var_regex = params_train_regularization_variable_regex
        self._l2_weight_decay = params.weight_decay

        # Checkpoint restoration.
        self._checkpoint_prefix = ""
        self._checkpoint_path = params.get("backbone_checkpoint", None)

    @abc.abstractmethod
    def build_outputs(self, inputs, is_training):
        """Build the graph of the forward path."""

    @abc.abstractmethod
    def build_model(self, weights, is_training):
        """Build the model object."""

    @abc.abstractmethod
    def build_loss_fn(self, keras_model, compression_loss_fn):
        """Build the model loss."""

    def post_processing(self, labels, outputs):
        """Post-processing function."""
        return labels, outputs

    def model_outputs(self, inputs, is_training):
        """Build the model outputs."""
        return self.build_outputs(inputs, is_training)

    def make_filter_trainable_variables_fn(self):
        """Creates a function for filtering trainable varialbes."""
        return _make_filter_trainable_variables_fn(self._frozen_variable_prefix)

    def weight_decay_loss(self, trainable_variables):
        reg_variables = [
            v
            for v in trainable_variables
            if self._regularization_var_regex is None or re.match(self._regularization_var_regex, v.name)
        ]

        return self._l2_weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in reg_variables])

    def make_restore_checkpoint_fn(self):
        """Returns scaffold function to restore parameters from v1 checkpoint."""
        skip_regex = None
        return checkpoint_utils.make_restore_checkpoint_fn(
            self._checkpoint_path, prefix=self._checkpoint_prefix, skip_regex=skip_regex
        )

    def eval_metrics(self):
        """Returns tuple of metric function and its inputs for evaluation."""
        raise NotImplementedError("Unimplemented eval_metrics")
