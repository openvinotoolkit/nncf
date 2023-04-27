:py:mod:`nncf.tensorflow`
=========================

.. py:module:: nncf.tensorflow

.. autoapi-nested-parse::

   Copyright (c) 2023 Intel Corporation
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.



Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   pruning/index.rst
   quantization/index.rst
   sparsity/index.rst




Functions
~~~~~~~~~

.. autoapisummary::

   nncf.tensorflow.create_compressed_model
   nncf.tensorflow.create_compression_callbacks
   nncf.tensorflow.register_default_init_args



.. py:function:: create_compressed_model(model, config, compression_state = None)

   The main function used to produce a model ready for compression fine-tuning
   from an original TensorFlow Keras model and a configuration object.

   :param model: The original model. Should have its parameters already loaded
       from a checkpoint or another source.
   :param config: A configuration object used to determine the exact compression
       modifications to be applied to the model.
   :type config: nncf.NNCFConfig
   :param compression_state: compression state to unambiguously restore the compressed model.
       Includes builder and controller states. If it is specified, trainable parameter initialization will be skipped
       during building.
   :return: A tuple (compression_ctrl, compressed_model) where
       - compression_ctrl: The controller of the compression algorithm.
       - compressed_model: The model with additional modifications
           necessary to enable algorithm-specific compression during fine-tuning.


.. py:function:: create_compression_callbacks(compression_ctrl, log_tensorboard=True, log_text=True, log_dir=None)


.. py:function:: register_default_init_args(nncf_config, data_loader, batch_size, device = None)

   Register extra structures in the NNCFConfig. Initialization of some
   compression algorithms requires certain extra structures.

   :param nncf_config: An instance of the NNCFConfig class without extra structures.
   :param data_loader: Dataset used for initialization.
   :param batch_size: Batch size used for initialization.
   :param device: Device to perform initialization. If `device` is `None` then the device
       of the model parameters will be used.
   :return: An instance of the NNCFConfig class with extra structures.


