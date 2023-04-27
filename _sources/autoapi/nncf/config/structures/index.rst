:py:mod:`nncf.config.structures`
================================

.. py:module:: nncf.config.structures

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


