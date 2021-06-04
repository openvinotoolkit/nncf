"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from tensorflow.python.keras.engine import training_utils
from tensorflow.python.profiler import trace
from tensorflow.python.eager import context
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.engine import data_adapter

from nncf import AdaptiveCompressionTrainingLoop
from beta.nncf.tensorflow.accuracy_aware_training.runner import TFAccuracyAwareTrainingRunner as \
        AccuracyAwareTrainingRunner


def accuracy_aware_fit(cls_instance, train_dataset, compression_ctrl,
                       nncf_config, callbacks, initial_epoch, steps_per_epoch,
                       tensorboard_writer, log_dir, uncompressed_model_accuracy,
                       validation_data, validation_steps, result_dict_to_val_metric_fn, **kwargs):

    def train_epoch_fn(compression_ctrl, model, epoch):
        model.reset_metrics()

        with model.distribute_strategy.scope(), \
            training_utils.RespectCompiledTrainableState(model):
            # pylint: disable=protected-access
            data_handler = data_adapter.DataHandler(
                x=train_dataset,
                y=None,
                sample_weight=None,
                batch_size=None,
                steps_per_epoch=steps_per_epoch,
                initial_epoch=initial_epoch,
                epochs=1,
                shuffle=True,
                class_weight=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                model=model,
                steps_per_execution=model._steps_per_execution)

        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks_container = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                model=model,
                epochs=1,
                verbose=1,
                add_progbar=True,
                steps=data_handler.inferred_steps
            )

        if model.train_function is None:
            model.train_function = model.make_train_function()
        _, iterator = next(data_handler.enumerate_epochs())

        callbacks_container.on_epoch_begin(epoch)
        with data_handler.catch_stop_iteration():
            for step in data_handler.steps():
                with trace.Trace(
                    'train',
                    epoch_num=epoch,
                    step_num=step,
                    batch_size=None,
                    _r=1):
                    callbacks_container.on_train_batch_begin(step)

                    tmp_logs = model.train_function(iterator)
                    if data_handler.should_sync:
                        context.async_wait()
                    logs = tmp_logs
                    end_step = step + data_handler.step_increment
                    callbacks_container.on_train_batch_end(end_step, logs)
                    if model.stop_training:
                        break

    def validate_fn(model, epoch=None):
        val_x, val_y, val_sample_weight = (
            data_adapter.unpack_x_y_sample_weight(validation_data))
        val_logs = model.evaluate(
            x=val_x,
            y=val_y,
            sample_weight=val_sample_weight,
            batch_size=None,
            steps=validation_steps,
            callbacks=callbacks,
            return_dict=True)
        return result_dict_to_val_metric_fn(val_logs)


    cls_instance.original_model_accuracy = uncompressed_model_accuracy
    # instantiate and run accuracy-aware training loop
    acc_aware_training_loop = AdaptiveCompressionTrainingLoop(nncf_config, compression_ctrl,
                                                              runner_cls=AccuracyAwareTrainingRunner)
    cls_instance = acc_aware_training_loop.run(cls_instance,
                                               train_epoch_fn=train_epoch_fn,
                                               validate_fn=validate_fn,
                                               tensorboard_writer=tensorboard_writer,
                                               log_dir=log_dir)
