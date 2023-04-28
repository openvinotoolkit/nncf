:orphan:

:py:mod:`nncf.config.structures`
================================

.. py:module:: nncf.config.structures



Classes
~~~~~~~

.. autoapisummary::

   nncf.config.structures.QuantizationRangeInitArgs
   nncf.config.structures.BNAdaptationInitArgs
   nncf.config.structures.ModelEvaluationArgs




.. py:class:: QuantizationRangeInitArgs(data_loader, device = None)

   Bases: :py:obj:`NNCFExtraConfigStruct`

   Stores additional arguments for quantization range initialization algorithms.


.. py:class:: BNAdaptationInitArgs(data_loader, device = None)

   Bases: :py:obj:`NNCFExtraConfigStruct`

   Stores additional arguments for batchnorm statistics adaptation algorithm.


.. py:class:: ModelEvaluationArgs(eval_fn)

   Bases: :py:obj:`NNCFExtraConfigStruct`

   This is the class from which all extra structures that define additional
   NNCFConfig arguments inherit.


