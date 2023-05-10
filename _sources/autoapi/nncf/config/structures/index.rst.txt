:py:mod:`nncf.config.structures`
================================

.. py:module:: nncf.config.structures

.. autoapi-nested-parse::

   Structures for passing live Python objects into NNCF algorithms.




Classes
~~~~~~~

.. autoapisummary::

   nncf.config.structures.QuantizationRangeInitArgs
   nncf.config.structures.BNAdaptationInitArgs
   nncf.config.structures.ModelEvaluationArgs




.. py:class:: QuantizationRangeInitArgs(data_loader, device = None)

   Bases: :py:obj:`NNCFExtraConfigStruct`

   Stores additional arguments for quantization range initialization algorithms.

   :param data_loader: Provides an iterable over the given dataset.
   :param device: Device to perform initialization. If `device` is `None`
       then the device of the model parameters will be used.


.. py:class:: BNAdaptationInitArgs(data_loader, device = None)

   Bases: :py:obj:`NNCFExtraConfigStruct`

   Stores additional arguments for batchnorm statistics adaptation algorithm.

   :param data_loader: Provides an iterable over the given dataset.
   :param device: Device to perform initialization. If `device` is `None`
       then the device of the model parameters will be used.


.. py:class:: ModelEvaluationArgs(eval_fn)

   Bases: :py:obj:`NNCFExtraConfigStruct`

   Stores additional arguments for running the model in the evaluation mode, should this be required for an algorithm.

   :param eval_fn: A function accepting a single argument - the model object - and returning the model's metric on
       the evaluation split of the dataset corresponding to the model.


