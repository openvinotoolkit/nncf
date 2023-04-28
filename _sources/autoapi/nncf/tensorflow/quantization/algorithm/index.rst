:orphan:

:py:mod:`nncf.tensorflow.quantization.algorithm`
================================================

.. py:module:: nncf.tensorflow.quantization.algorithm



Classes
~~~~~~~

.. autoapisummary::

   nncf.tensorflow.quantization.algorithm.QuantizationController




.. py:class:: QuantizationController(target_model, config, op_names)

   Bases: :py:obj:`nncf.common.compression.BaseCompressionAlgorithmController`

   Contains the implementation of the basic functionality of the compression controller.

   .. py:method:: strip_model(model, do_copy = False)

      Strips auxiliary layers that were used for the model compression, as it's
      only needed for training. The method is used before exporting the model
      in the target format.

      :param model: The compressed model.
      :param do_copy: Modify copy of the model, defaults to False.
      :return: The stripped model.


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



