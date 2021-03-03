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

# TODO: Unify CompressionLoss
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
