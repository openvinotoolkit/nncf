:py:mod:`nncf.tensorflow.pruning.base_algorithm`
================================================

.. py:module:: nncf.tensorflow.pruning.base_algorithm



Classes
~~~~~~~

.. autoapisummary::

   nncf.tensorflow.pruning.base_algorithm.BasePruningAlgoController




.. py:class:: BasePruningAlgoController(target_model, op_names, prunable_types, pruned_layer_groups_info, config)

   Bases: :py:obj:`nncf.common.compression.BaseCompressionAlgorithmController`, :py:obj:`abc.ABC`

   Base class for TF pruning algorithm controllers.

   .. py:method:: strip_model(model, do_copy = False)

      Strips auxiliary layers that were used for the model compression, as it's
      only needed for training. The method is used before exporting the model
      in the target format.

      :param model: The compressed model.
      :param do_copy: Modify copy of the model, defaults to False.
      :return: The stripped model.



