:py:mod:`nncf.tensorflow.sparsity.magnitude.algorithm`
======================================================

.. py:module:: nncf.tensorflow.sparsity.magnitude.algorithm

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




Classes
~~~~~~~

.. autoapisummary::

   nncf.tensorflow.sparsity.magnitude.algorithm.MagnitudeSparsityController




.. py:class:: MagnitudeSparsityController(target_model, config, op_names)

   Bases: :py:obj:`nncf.tensorflow.sparsity.base_algorithm.BaseSparsityController`

   Serves as a handle to the additional modules, parameters and hooks inserted
   into the original uncompressed model in order to enable algorithm-specific compression.
   Hosts entities that are to be used during the training process, such as compression scheduler and
   compression loss.

   .. py:method:: freeze(freeze = True)

      Freezes all sparsity masks. Sparsity masks will not be trained after calling this method.


   .. py:method:: set_sparsity_level(sparsity_level, run_batchnorm_adaptation = False)

      Sets the sparsity level that should be applied to the model's weights.

      :param sparsity_level: Sparsity level that should be applied to the model's weights.


   .. py:method:: disable_scheduler()

      Disables current compression scheduler during training by changing
      it to a dummy one that does not change the compression rate.


   .. py:method:: statistics(quickly_collected_only = False)

      Returns a `Statistics` class instance that contains compression algorithm statistics.

      :param quickly_collected_only: Enables collection of the statistics that
          don't take too much time to compute. Can be helpful for the case when
          need to keep track of statistics on each training batch/step/iteration.
      :return: A `Statistics` class instance that contains compression algorithm statistics.


   .. py:method:: compression_stage()

      Returns the compression stage. Should be used on saving best checkpoints
      to distinguish between uncompressed, partially compressed, and fully
      compressed models.

      :return: The compression stage of the target model.



