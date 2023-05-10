:py:mod:`nncf.common.initialization.dataloader`
===============================================

.. py:module:: nncf.common.initialization.dataloader

.. autoapi-nested-parse::

   Interface for user-defined data usage during the compression algorithm initialization process.




Classes
~~~~~~~

.. autoapisummary::

   nncf.common.initialization.dataloader.NNCFDataLoader




.. py:class:: NNCFDataLoader

   Bases: :py:obj:`abc.ABC`

   Wraps a custom data source.

   .. py:property:: batch_size
      :type: int
      :abstractmethod:

      Returns the number of elements return per iteration.

      :return: A number of elements return per iteration.


   .. py:method:: __iter__()
      :abstractmethod:

      Creates an iterator for the elements of a custom data source.
      The returned iterator implements the Python Iterator protocol.

      :return: An iterator for the elements of a custom data source.



