:py:mod:`nncf.common.sparsity.statistics`
=========================================

.. py:module:: nncf.common.sparsity.statistics



Classes
~~~~~~~

.. autoapisummary::

   nncf.common.sparsity.statistics.SparsifiedLayerSummary
   nncf.common.sparsity.statistics.SparsifiedModelStatistics
   nncf.common.sparsity.statistics.MagnitudeSparsityStatistics
   nncf.common.sparsity.statistics.ConstSparsityStatistics
   nncf.common.sparsity.statistics.RBSparsityStatistics
   nncf.common.sparsity.statistics.MovementSparsityStatistics




.. py:class:: SparsifiedLayerSummary(name, weight_shape, sparsity_level, weight_percentage)

   Contains information about the sparsified layer.

   :param name: Layer's name.
   :param weight_shape: Weight's shape.
   :param sparsity_level: Sparsity level of the sparsified layer.
   :param weight_percentage: Proportion of the layer's weights in the whole model.


.. py:class:: SparsifiedModelStatistics(sparsity_level, sparsity_level_for_layers, sparsified_layers_summary)

   Bases: :py:obj:`nncf.api.statistics.Statistics`

   Contains statistics of the sparsified model.

   :param sparsity_level: Sparsity level of the whole model.
   :param sparsity_level_for_layers: Sparsity level of all sparsified layers
     (i.e. layers for which the algorithm was applied).
   :param sparsified_layers_summary: Detailed summary for the sparsified layers.

   .. py:method:: to_str()

      Returns a representation of the statistics as a human-readable string.



.. py:class:: MagnitudeSparsityStatistics(model_statistics, thresholds, target_sparsity_level)

   Bases: :py:obj:`nncf.api.statistics.Statistics`

   Contains statistics of the magnitude sparsity algorithm.

   :param model_statistics: Statistics of the sparsified model.
   :param thresholds: List of the sparsity thresholds.
   :param target_sparsity_level: A target level of the sparsity for the algorithm for the current epoch.

   .. py:method:: to_str()

      Returns a representation of the statistics as a human-readable string.



.. py:class:: ConstSparsityStatistics(model_statistics)

   Bases: :py:obj:`nncf.api.statistics.Statistics`

   Contains statistics of the const sparsity algorithm.

   :param model_statistics: Statistics of the sparsified model.

   .. py:method:: to_str()

      Returns a representation of the statistics as a human-readable string.



.. py:class:: RBSparsityStatistics(model_statistics, target_sparsity_level, mean_sparse_prob)

   Bases: :py:obj:`nncf.api.statistics.Statistics`

   Contains statistics of the RB-sparsity algorithm.

   :param model_statistics: Statistics of the sparsified model.
   :param target_sparsity_level: A target level of the sparsity for the algorithm for the current epoch.
   :param mean_sparse_prob: The probability that one weight will be zeroed.

   .. py:method:: to_str()

      Returns a representation of the statistics as a human-readable string.



.. py:class:: MovementSparsityStatistics(model_statistics, importance_threshold, importance_regularization_factor)

   Bases: :py:obj:`nncf.api.statistics.Statistics`

   Contains statistics of the movement-sparsity algorithm.

   :param model_statistics: Statistics of the sparsified model.
   :param importance_threshold: Importance threshold for sparsity binary mask.
   :param importance_regularization_factor: Penalty factor of importance score.

   .. py:method:: to_str()

      Returns a representation of the statistics as a human-readable string.



