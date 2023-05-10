:py:mod:`nncf.common.pruning.schedulers`
========================================

.. py:module:: nncf.common.pruning.schedulers



Classes
~~~~~~~

.. autoapisummary::

   nncf.common.pruning.schedulers.PruningScheduler




.. py:class:: PruningScheduler(controller, params)

   Bases: :py:obj:`nncf.common.schedulers.BaseCompressionScheduler`

   This is the class from which all pruning schedulers inherit.

   A pruning scheduler is an object which specifies the pruning
   level at each training epoch. It involves a scheduling algorithm,
   defined in the `_calculate_pruning_level()` method and a state
   (some parameters required for current pruning level calculation)
   defined in the `__init__()` method.

   :param controller: Pruning algorithm controller.
   :param params: Parameters of the scheduler in the JSON-like dictionary form. Passed as-is from the corresponding
     section of the NNCF config file .json section (https://openvinotoolkit.github.io/nncf/schema).

   .. py:property:: current_pruning_level
      :type: float

      Returns pruning level for the `current_epoch`.

      :return: Current sparsity level.


   .. py:method:: epoch_step(next_epoch = None)

      Should be called at the beginning of each training epoch to prepare
      the pruning method to continue training the model in the `next_epoch`.

      :param next_epoch: The epoch index for which the pruning scheduler
          will update the state of the pruning method.


   .. py:method:: step(next_step = None)

      Should be called at the beginning of each training step to prepare
      the pruning method to continue training the model in the `next_step`.

      :param next_step: The global step index for which the pruning scheduler
          will update the state of the pruning method.



