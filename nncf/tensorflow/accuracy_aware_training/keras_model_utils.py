"""
 Copyright (c) 2022 Intel Corporation
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
import copy

from tensorflow.python.keras.engine import training_utils
from tensorflow.python.profiler import trace
from tensorflow.python.eager import context
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.engine import data_adapter

from nncf.common.accuracy_aware_training import create_accuracy_aware_training_loop


def accuracy_aware_fit(cls_instance, train_dataset, compression_ctrl,
                       nncf_config, callbacks, initial_epoch, uncompressed_model_accuracy,
                       steps_per_epoch=None, batch_size=None, tensorboard_writer=None,
                       log_dir=None, validation_data=None, validation_steps=None,
                       result_dict_to_val_metric_fn=None, **kwargs):
    if result_dict_to_val_metric_fn is None:
        result_dict_to_val_metric_fn = lambda metric: metric

    with cls_instance.distribute_strategy.scope(), \
        training_utils.RespectCompiledTrainableState(cls_instance):
        # pylint: disable=protected-access
        data_handler = data_adapter.DataHandler(
            x=train_dataset,
            y=None,
            sample_weight=None,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            initial_epoch=initial_epoch,
            epochs=1,
            shuffle=True,
            class_weight=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            model=cls_instance,
            steps_per_execution=cls_instance._steps_per_execution)

        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                model=cls_instance,
                epochs=1,
                verbose=1,
                add_progbar=True,
                steps=data_handler.inferred_steps
                )

    def train_epoch_fn(compression_ctrl, model, epoch):
        model.reset_metrics()

        if model.train_function is None:
            model.train_function = model.make_train_function()
        _, iterator = next(data_handler.enumerate_epochs())

        callbacks.on_epoch_begin(epoch)
        with data_handler.catch_stop_iteration():
            for step in data_handler.steps():
                with trace.Trace(
                    'train',
                    epoch_num=epoch,
                    step_num=step,
                    batch_size=None,
                    _r=1):
                    callbacks.on_train_batch_begin(step)
                    tmp_logs = model.train_function(iterator)
                    if data_handler.should_sync:
                        context.async_wait()
                    logs = tmp_logs
                    end_step = step + data_handler.step_increment
                    callbacks.on_train_batch_end(end_step, logs)
                    if model.stop_training:
                        break

        if logs is None:
            raise ValueError('Expect x to be a non-empty array or dataset.')
        epoch_logs = copy.copy(logs)
        callbacks.on_epoch_end(epoch, epoch_logs)

    if validation_data is None:
        validation_data = train_dataset

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


    callbacks.on_train_begin()
    cls_instance.original_model_accuracy = uncompressed_model_accuracy
    acc_aware_training_loop = create_accuracy_aware_training_loop(nncf_config, compression_ctrl)
    cls_instance = acc_aware_training_loop.run(cls_instance,
                                               train_epoch_fn=train_epoch_fn,
                                               validate_fn=validate_fn,
                                               tensorboard_writer=tensorboard_writer,
                                               log_dir=log_dir)
    callbacks.on_train_end()
