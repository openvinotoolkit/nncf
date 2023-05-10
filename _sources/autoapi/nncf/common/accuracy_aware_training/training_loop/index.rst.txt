:py:mod:`nncf.common.accuracy_aware_training.training_loop`
===========================================================

.. py:module:: nncf.common.accuracy_aware_training.training_loop

.. autoapi-nested-parse::

   Implementations of training loops to be used for accuracy aware training.




Classes
~~~~~~~

.. autoapisummary::

   nncf.common.accuracy_aware_training.training_loop.TrainingLoop
   nncf.common.accuracy_aware_training.training_loop.BaseEarlyExitCompressionTrainingLoop
   nncf.common.accuracy_aware_training.training_loop.EarlyExitCompressionTrainingLoop
   nncf.common.accuracy_aware_training.training_loop.AdaptiveCompressionTrainingLoop




.. py:class:: TrainingLoop

   Bases: :py:obj:`abc.ABC`

   The training loop object that launches the training process via the `run` method.

   .. py:property:: statistics
      :type: nncf.common.accuracy_aware_training.statistics.TrainingLoopStatistics
      :abstractmethod:

      Returns statistics of the compressed model.


   .. py:method:: run(model, train_epoch_fn, validate_fn, configure_optimizers_fn = None, dump_checkpoint_fn = None, load_checkpoint_fn = None, early_stopping_fn = None, tensorboard_writer = None, log_dir = None, update_learning_rate_fn = None)
      :abstractmethod:

      Implements the custom logic to run a training loop for model fine-tuning by using the provided
      `train_epoch_fn`, `validate_fn` and `configure_optimizers_fn` methods.

      :param model: The model instance before fine-tuning
      :param train_epoch_fn: a callback to fine-tune the model for a single epoch
      :param validate_fn: a callback to evaluate the model on the validation dataset
      :param configure_optimizers_fn: a callback to instantiate an optimizer and a learning rate scheduler
      :param dump_checkpoint_fn: a callback to dump a checkpoint
      :param load_checkpoint_fn: a callback to load a checkpoint
      :param early_stopping_fn: a callback to check for an early stopping condition
      :param tensorboard_writer: The tensorboard object to be used for logging.
      :param log_dir: The path to be used for logging and checkpoint saving.
      :param update_learning_rate_fn: The callback to update the learning rate after each epoch
        of the training loop.
      :return: The fine-tuned model.



.. py:class:: BaseEarlyExitCompressionTrainingLoop(compression_controller)

   Bases: :py:obj:`TrainingLoop`, :py:obj:`abc.ABC`

   Base class to generalize functionality of derived training loop classes.

   .. py:property:: statistics
      :type: nncf.common.accuracy_aware_training.statistics.TrainingLoopStatistics

      Returns statistics of the compressed model.


   .. py:method:: run(model, train_epoch_fn, validate_fn, configure_optimizers_fn = None, dump_checkpoint_fn = None, load_checkpoint_fn = None, early_stopping_fn = None, tensorboard_writer = None, log_dir = None, update_learning_rate_fn = None)

      Implements the custom logic to run a training loop for model fine-tuning by using the provided
      `train_epoch_fn`, `validate_fn` and `configure_optimizers_fn` methods.

      :param model: The model instance before fine-tuning
      :param train_epoch_fn: a callback to fine-tune the model for a single epoch
      :param validate_fn: a callback to evaluate the model on the validation dataset
      :param configure_optimizers_fn: a callback to instantiate an optimizer and a learning rate scheduler
      :param dump_checkpoint_fn: a callback to dump a checkpoint
      :param load_checkpoint_fn: a callback to load a checkpoint
      :param early_stopping_fn: a callback to check for an early stopping condition
      :param tensorboard_writer: The tensorboard object to be used for logging.
      :param log_dir: The path to be used for logging and checkpoint saving.
      :param update_learning_rate_fn: The callback to update the learning rate after each epoch
        of the training loop.
      :return: The fine-tuned model.



.. py:class:: EarlyExitCompressionTrainingLoop(nncf_config, compression_controller, uncompressed_model_accuracy, lr_updates_needed = True, verbose = True, dump_checkpoints = True)

   Bases: :py:obj:`BaseEarlyExitCompressionTrainingLoop`

   Training loop that does not modify compression parameters and exits as soon as (and if) the accuracy drop criterion
   is reached.

   :param nncf_config: The configuration object.
   :type nncf_config: nncf.NNCFConfig
   :param compression_controller: The controller for the compression algorithm that is currently applied to the model
       to be trained.
   :param uncompressed_model_accuracy: The uncompressed model accuracy, measured outside of this training loop to
       serve as the point of reference for fine-tuning the compressed model.
   :param lr_updates_needed:
   :param verbose: Whether to post additional data to TensorBoard.
   :param dump_checkpoints: If true, will dump all checkpoints obtained during the training process, otherwise will
     only keep the best checkpoint (accuracy-wise).


.. py:class:: AdaptiveCompressionTrainingLoop(nncf_config, compression_controller, uncompressed_model_accuracy, lr_updates_needed = True, verbose = True, minimal_compression_rate = 0.0, maximal_compression_rate = 0.95, dump_checkpoints = True)

   Bases: :py:obj:`BaseEarlyExitCompressionTrainingLoop`

   A training loop that automatically adjusts compression rate to reach maximum compression within accuracy budget.

   :param nncf_config: The configuration object.
   :type nncf_config: nncf.NNCFConfig
   :param compression_controller: The controller for the compression algorithm that is currently applied to the model
       to be trained.
   :param uncompressed_model_accuracy: The uncompressed model accuracy, measured outside of this training loop to
       serve as the point of reference for fine-tuning the compressed model.
   :param lr_updates_needed:
   :param verbose: Whether to post additional data to TensorBoard.
   :param minimal_compression_rate: Sets the minimal compression rate to be considered during the training loop.
   :param maximal_compression_rate: Sets the maximal compression rate to be considered during the training loop.
   :param dump_checkpoints: If true, will dump all checkpoints obtained during the training process, otherwise will
     only keep the best checkpoint (accuracy-wise).

   .. py:method:: run(model, train_epoch_fn, validate_fn, configure_optimizers_fn = None, dump_checkpoint_fn = None, load_checkpoint_fn = None, early_stopping_fn = None, tensorboard_writer = None, log_dir = None, update_learning_rate_fn = None)

      Implements the custom logic to run a training loop for model fine-tuning by using the provided
      `train_epoch_fn`, `validate_fn` and `configure_optimizers_fn` methods.

      :param model: The model instance before fine-tuning
      :param train_epoch_fn: a callback to fine-tune the model for a single epoch
      :param validate_fn: a callback to evaluate the model on the validation dataset
      :param configure_optimizers_fn: a callback to instantiate an optimizer and a learning rate scheduler
      :param dump_checkpoint_fn: a callback to dump a checkpoint
      :param load_checkpoint_fn: a callback to load a checkpoint
      :param early_stopping_fn: a callback to check for an early stopping condition
      :param tensorboard_writer: The tensorboard object to be used for logging.
      :param log_dir: The path to be used for logging and checkpoint saving.
      :param update_learning_rate_fn: The callback to update the learning rate after each epoch
        of the training loop.
      :return: The fine-tuned model.



