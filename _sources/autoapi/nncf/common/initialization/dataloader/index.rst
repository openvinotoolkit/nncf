:orphan:

:py:mod:`nncf.common.initialization.dataloader`
===============================================

.. py:module:: nncf.common.initialization.dataloader



Classes
~~~~~~~

.. autoapisummary::

   nncf.common.initialization.dataloader.NNCFDataLoader




.. py:class:: NNCFDataLoader

   Bases: :py:obj:`abc.ABC`

   Wraps a custom data source.

   .. py:method:: __iter__()
      :abstractmethod:

      Creates an iterator for the elements of a custom data source.
      The returned iterator implements the Python Iterator protocol.

      :return: An iterator for the elements of a custom data source.



