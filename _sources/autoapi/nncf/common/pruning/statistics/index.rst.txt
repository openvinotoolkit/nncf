:py:mod:`nncf.common.pruning.statistics`
========================================

.. py:module:: nncf.common.pruning.statistics



Classes
~~~~~~~

.. autoapisummary::

   nncf.common.pruning.statistics.FilterPruningStatistics




.. py:class:: FilterPruningStatistics(model_statistics, current_pruning_level, target_pruning_level, prune_flops)

   Bases: :py:obj:`nncf.api.statistics.Statistics`

   Contains statistics of the filter pruning algorithm.

   :param model_statistics: Statistics of the pruned model.
   :param current_pruning_level: A current level of the pruning for the algorithm for the current epoch.
   :param target_pruning_level: A target level of the pruning for the algorithm.
   :param prune_flops: Is pruning algo sets flops pruning level or not (filter pruning level).

   .. py:method:: to_str()

      Returns a representation of the statistics as a human-readable string.



