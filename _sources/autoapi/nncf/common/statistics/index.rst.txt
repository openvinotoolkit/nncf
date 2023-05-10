:py:mod:`nncf.common.statistics`
================================

.. py:module:: nncf.common.statistics



Classes
~~~~~~~

.. autoapisummary::

   nncf.common.statistics.NNCFStatistics




.. py:class:: NNCFStatistics

   Bases: :py:obj:`nncf.api.statistics.Statistics`

   Groups statistics for all available NNCF compression algorithms.
   Statistics are present only if the algorithm has been started.

   .. py:property:: magnitude_sparsity
      :type: Optional[nncf.common.sparsity.statistics.MagnitudeSparsityStatistics]

      Returns statistics of the magnitude sparsity algorithm. If statistics
      have not been collected, `None` will be returned.

      :return: Instance of the `MagnitudeSparsityStatistics` class.


   .. py:property:: rb_sparsity
      :type: Optional[nncf.common.sparsity.statistics.RBSparsityStatistics]

      Returns statistics of the RB-sparsity algorithm. If statistics
      have not been collected, `None` will be returned.

      :return: Instance of the `RBSparsityStatistics` class.


   .. py:property:: movement_sparsity
      :type: Optional[nncf.common.sparsity.statistics.MovementSparsityStatistics]

      Returns statistics of the movement sparsity algorithm. If statistics
      have not been collected, `None` will be returned.

      :return: Instance of the `MovementSparsityStatistics` class.


   .. py:property:: const_sparsity
      :type: Optional[nncf.common.sparsity.statistics.ConstSparsityStatistics]

      Returns statistics of the const sparsity algorithm. If statistics
      have not been collected, `None` will be returned.

      :return: Instance of the `ConstSparsityStatistics` class.


   .. py:property:: quantization
      :type: Optional[nncf.common.quantization.statistics.QuantizationStatistics]

      Returns statistics of the quantization algorithm. If statistics
      have not been collected, `None` will be returned.

      :return: Instance of the `QuantizationStatistics` class.


   .. py:property:: filter_pruning
      :type: Optional[nncf.common.pruning.statistics.FilterPruningStatistics]

      Returns statistics of the filter pruning algorithm. If statistics
      have not been collected, `None` will be returned.

      :return: Instance of the `FilterPruningStatistics` class.


   .. py:method:: register(algorithm_name, stats)

      Registers statistics for the algorithm.

      :param algorithm_name: Name of the algorithm. Should be one of the following
          * magnitude_sparsity
          * rb_sparsity
          * const_sparsity
          * quantization
          * filter_pruning
          * binarization

      :param stats: Statistics of the algorithm.


   .. py:method:: to_str()

      Calls `to_str()` method for all registered statistics of the algorithm and returns
      a sum-up string.

      :return: A representation of the NNCF statistics as a human-readable string.



