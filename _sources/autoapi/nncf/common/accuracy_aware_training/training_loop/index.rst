:orphan:

:py:mod:`nncf.common.accuracy_aware_training.training_loop`
===========================================================

.. py:module:: nncf.common.accuracy_aware_training.training_loop



Classes
~~~~~~~

.. autoapisummary::

   nncf.common.accuracy_aware_training.training_loop.EarlyExitCompressionTrainingLoop
   nncf.common.accuracy_aware_training.training_loop.AdaptiveCompressionTrainingLoop




.. py:class:: EarlyExitCompressionTrainingLoop(nncf_config, compression_controller, uncompressed_model_accuracy, lr_updates_needed = True, verbose = True, dump_checkpoints = True)

   Bases: :py:obj:`BaseEarlyExitCompressionTrainingLoop`

   Adaptive compression training loop allows an accuracy-aware training process
   to reach the maximal accuracy drop
   (the maximal allowed accuracy degradation criterion is satisfied).


.. py:class:: AdaptiveCompressionTrainingLoop(nncf_config, compression_controller, uncompressed_model_accuracy, lr_updates_needed=True, verbose=True, minimal_compression_rate=0.0, maximal_compression_rate=0.95, dump_checkpoints=True)

   Bases: :py:obj:`BaseEarlyExitCompressionTrainingLoop`

   Adaptive compression training loop allows an accuracy-aware training process whereby
   the compression rate is automatically varied during training to reach the maximal
   possible compression rate with a positive accuracy budget
   (the maximal allowed accuracy degradation criterion is satisfied).

   .. py:method:: run(model, train_epoch_fn, validate_fn, configure_optimizers_fn=None, dump_checkpoint_fn=None, load_checkpoint_fn=None, early_stopping_fn=None, tensorboard_writer=None, log_dir=None, update_learning_rate_fn=None)

      Implements the custom logic to run a training loop for model fine-tuning
      by using the provided `train_epoch_fn`, `validate_fn` and `configure_optimizers_fn` methods.
      The passed methods are registered in the `TrainingRunner` instance and the training logic
      is implemented by calling the corresponding `TrainingRunner` methods

      :param model: The model instance before fine-tuning
      :param train_epoch_fn: a method to fine-tune the model for a single epoch
      (to be called inside the `train_epoch` of the TrainingRunner)
      :param validate_fn: a method to evaluate the model on the validation dataset
      (to be called inside the `train_epoch` of the TrainingRunner)
      :param configure_optimizers_fn: a method to instantiate an optimizer and a learning
      rate scheduler (to be called inside the `configure_optimizers` of the TrainingRunner)
      :param dump_checkpoint_fn: a method to dump a checkpoint
      :param load_checkpoint_fn: a method to load a checkpoint
      :param early_stopping_fn: a method to check for an early stopping condition
      :return: The fine-tuned model



