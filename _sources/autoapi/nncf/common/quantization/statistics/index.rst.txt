:py:mod:`nncf.common.quantization.statistics`
=============================================

.. py:module:: nncf.common.quantization.statistics



Classes
~~~~~~~

.. autoapisummary::

   nncf.common.quantization.statistics.QuantizationStatistics




.. py:class:: QuantizationStatistics(wq_counter, aq_counter, num_wq_per_bitwidth, num_aq_per_bitwidth, ratio_of_enabled_quantizations)

   Bases: :py:obj:`nncf.api.statistics.Statistics`

   Contains statistics of the quantization algorithm. These statistics include:

   * Information about the share of the quantization, such as:

     * Percentage of symmetric/asymmetric/per-channel/per-tensor weight quantizers relative to the number of placed
       weight quantizers.
     * Percentage of symmetric/asymmetric/per-channel/per-tensor non-weight quantizers relative to the number of
       placed non weight quantizers.
     * Percentage of weight quantizers and non-weight quantizers for each precision relative to the number
       of potential quantizers/placed quantizers.

   * Information about the distribution of the bitwidth of the quantizers.
   * Ratio of enabled quantization.

   .. note:: The maximum possible number of potential quantizers depends on the presence of ignored scopes and the
     mode of quantizer setup that is used at the time of collecting the metric.

   :param wq_counter: Weight quantizers counter.
   :param aq_counter: Activation quantizers counter.
   :param num_wq_per_bitwidth: Number of weight quantizers per bit width.
   :param num_aq_per_bitwidth: Number of activation quantizers per bit width.
   :param ratio_of_enabled_quantizations: Ratio of enabled quantizations.

   .. py:method:: to_str()

      Returns a representation of the statistics as a human-readable string.



