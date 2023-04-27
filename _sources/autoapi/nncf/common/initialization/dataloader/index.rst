:py:mod:`nncf.common.initialization.dataloader`
===============================================

.. py:module:: nncf.common.initialization.dataloader

.. autoapi-nested-parse::

   Copyright (c) 2023 Intel Corporation
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.




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



