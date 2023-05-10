:py:mod:`nncf.api.statistics`
=============================

.. py:module:: nncf.api.statistics



Classes
~~~~~~~

.. autoapisummary::

   nncf.api.statistics.Statistics




.. py:class:: Statistics

   Bases: :py:obj:`abc.ABC`

   Contains a collection of model- or compression-related data and provides a way for its human-readable
   representation.

   .. py:method:: to_str()
      :abstractmethod:

      Returns a representation of the statistics as a human-readable string.



