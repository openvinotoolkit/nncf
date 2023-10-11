:py:mod:`nncf.tensorflow.sparsity.magnitude.algorithm`
======================================================

.. py:module:: nncf.tensorflow.sparsity.magnitude.algorithm



Classes
~~~~~~~

.. autoapisummary::

   nncf.tensorflow.sparsity.magnitude.algorithm.MagnitudeSparsityController




.. py:class:: MagnitudeSparsityController(target_model, config, op_names)

   Bases: :py:obj:`nncf.tensorflow.sparsity.base_algorithm.BaseSparsityController`

   Controller class for magnitude sparsity in TF.

   .. py:property:: scheduler
      :type: nncf.api.compression.CompressionScheduler

      The compression scheduler for this particular algorithm combination.


   .. py:property:: loss
      :type: nncf.api.compression.CompressionLoss

      The compression loss for this particular algorithm combination.


   .. py:property:: compression_rate
      :type: float

      Returns a float compression rate value ranging from 0 to 1 (e.g. the sparsity level,
      or the ratio of filters pruned).


   .. py:method:: freeze(freeze = True)

      Freezes all sparsity masks. Sparsity masks will not be trained after calling this method.


   .. py:method:: set_sparsity_level(sparsity_level, run_batchnorm_adaptation = False)

      Sets the sparsity level that should be applied to the model's weights.

      :param sparsity_level: Sparsity level that should be applied to the model's weights.
      :param run_batchnorm_adaptation: Whether to run batchnorm adaptation after setting the sparsity level.


   .. py:method:: disable_scheduler()

      Disables current compression scheduler during training by changing it to a dummy one that does not change
      the compression rate.


   .. py:method:: statistics(quickly_collected_only = False)

      Returns a `Statistics` class instance that contains compression algorithm statistics.

      :param quickly_collected_only: Enables collection of the statistics that
          don't take too much time to compute. Can be helpful for the case when
          need to keep track of statistics on each training batch/step/iteration.


   .. py:method:: compression_stage()

      Returns the compression stage. Should be used on saving best checkpoints
      to distinguish between uncompressed, partially compressed, and fully
      compressed models.

      :return: The compression stage of the target model.



