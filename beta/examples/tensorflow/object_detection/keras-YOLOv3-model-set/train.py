#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrain the YOLO model for your own dataset.
"""
import os, time, random, argparse
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
# from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TerminateOnNaN, LambdaCallback
# from tensorflow_model_optimization.sparsity import keras as sparsity

from yolo3.loss import build_loss_fn
from yolo3.model import get_yolo3_model, get_yolo3_train_model, get_yolo3_train_model_custom
from yolo3.data import yolo3_data_generator_wrapper, get_dataset_builders # Yolo3DataGenerator
from common.utils import get_classes, get_anchors, get_dataset, optimize_tf_gpu
from common.model_utils import get_optimizer
# from common.callbacks import EvalCallBack, DatasetShuffleCallBack

from beta.examples.tensorflow.common.distributed import get_distribution_strategy
from beta.examples.tensorflow.common.logger import logger
from beta.examples.tensorflow.common.utils import Timer

# Try to enable Auto Mixed Precision on TF 2.0
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
optimize_tf_gpu(tf, K)


def main(args):
    annotation_file = args.annotation_file
    log_dir = os.path.join('logs', '000')
    classes_path = args.classes_path
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    anchors = get_anchors(args.anchors_path)
    # num_anchors = len(anchors)

    # callbacks for training process
    logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, mode='min', patience=10, verbose=1, cooldown=0, min_lr=1e-10)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='min')
    terminate_on_nan = TerminateOnNaN()

    callbacks=[logging, checkpoint, reduce_lr, early_stopping, terminate_on_nan]

    config = {}
    config['dataset'] = 'coco/2017'
    config['dataset_type'] = 'tfds'
    config['global_batch_size'] = args.batch_size
    config['dataset_dir'] = '/datasets/coco2017_tfds'
    config['model'] = 'YOLOv4'
    config['input_shape'] = args.model_image_size
    config['enhance_augment'] = args.enhance_augment
    config['anchors_path'] = args.anchors_path
    config['num_classes'] = num_classes
    config['multi_anchor_assign'] = args.multi_anchor_assign
    config['checkpoint_save_dir'] = 'logs'
    config['print_freq'] = 1

    num_devices = args.gpu_num

    strategy = get_distribution_strategy(config)

    print('Building dataset')
    builders = get_dataset_builders(config, strategy.num_replicas_in_sync)
    datasets = [builder.build() for builder in builders]
    train_builder, test_builder = builders
    train_dataset, test_dataset = datasets
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    # test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


    # import tensorflow_datasets as tfds
    # for ds_item in tfds.as_numpy(train_dataset):
    #     ds_item = ds_item[0]
    #     print('ds_item\n', type(ds_item)) # ds_item[0].shape
    #     break


    # get train&val dataset
    dataset = get_dataset(annotation_file)
    if args.val_annotation_file:
        val_dataset = get_dataset(args.val_annotation_file)
        num_train = len(dataset)
        num_val = len(val_dataset)
        dataset.extend(val_dataset)
    else:
        val_split = args.val_split
        num_val = int(len(dataset)*val_split)
        num_train = len(dataset) - num_val

    # assign multiscale interval
    if args.multiscale:
        rescale_interval = args.rescale_interval
    else:
        rescale_interval = -1  #Doesn't rescale

    # model input shape check
    input_shape = args.model_image_size
    assert (input_shape[0]%32 == 0 and input_shape[1]%32 == 0), 'model_image_size should be multiples of 32'


    # #################################### Pipeline testing ###############################
    # # Fix np.random.rand()*(b-a) in common/data_utils.py
    # # undo np.random.shuffle(annotation_lines) in data.py
    # # set self._num_preprocess_workers to one on data.py ???
    # data_generator = yolo3_data_generator_wrapper
    # import tensorflow_datasets as tfds
    # print('\nTFDS pipeline\n')
    #
    # for ds_item in tfds.as_numpy(train_dataset):
    #     ds_item = ds_item[0]
    #     break
    #
    # filenames = [item.decode("utf-8") for item in ds_item['filename']] # first 000000050124.jpg 000000271058.jpg
    # filename = filenames[0]
    # print('filename', filename)
    #
    # image_tfds = ds_item['image_input'][0]
    # out0 = ds_item['y_true_0'][0]
    # out1 = ds_item['y_true_1'][0]
    # out2 = ds_item['y_true_2'][0]
    #
    # print(image_tfds.shape, out0.shape, out1.shape, out2.shape)
    # print('image mean', image_tfds.mean())
    # print('image\n', image_tfds[100:102, 100:102, :])
    # print('out means', out1.mean(), out1.mean(), out2.mean())
    #
    #
    #
    #
    # print('\nOriginal pipeline\n')
    # dataset = dataset[:num_train]
    # print('founding the item in origin dataset...')
    # for item in dataset:
    #     if filename in item:
    #         print('Found the item', item)
    #         break
    # dataset.insert(0, item)
    # data_gen = data_generator(dataset, args.batch_size, input_shape, anchors, num_classes, args.enhance_augment,
    #                rescale_interval, multi_anchor_assign=args.multi_anchor_assign)
    #
    # out = next(data_gen)
    # image, out0, out1, out2 = out[0]
    # image = image[0]
    # out0 = out0[0]
    # out1 = out1[0]
    # out2 = out2[0]
    # print(image.shape, out0.shape, out1.shape, out2.shape)
    # print('image mean', image.mean())
    # print('image\n', image[100:102, 100:102, :])
    # print('out means', out0.mean(), out1.mean(), out2.mean())
    # ######################################################################




    # # Get and prepare the model
    # get_train_model = get_yolo3_train_model
    #
    # # prepare optimizer
    # optimizer = get_optimizer(args.optimizer, args.learning_rate, decay_type=None)
    #
    # # support multi-gpu training
    # if args.gpu_num >= 2:
    #     # devices_list=["/gpu:0", "/gpu:1"]
    #     devices_list=["/gpu:{}".format(n) for n in range(args.gpu_num)]
    #     strategy = tf.distribute.MirroredStrategy(devices=devices_list)
    #     print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    #     with strategy.scope():
    #         # get multi-gpu train model
    #         model = get_train_model(args.model_type, anchors, num_classes, weights_path=args.weights_path, optimizer=optimizer, label_smoothing=args.label_smoothing, elim_grid_sense=args.elim_grid_sense)
    # else:
    #     # get normal train model
    #     model = get_train_model(args.model_type, anchors, num_classes, weights_path=args.weights_path, optimizer=optimizer, label_smoothing=args.label_smoothing, elim_grid_sense=args.elim_grid_sense)
    #
    # model.summary()
    #
    # if args.decay_type:
    #     # rebuild optimizer to apply learning rate decay, only after
    #     # unfreeze all layers
    #     callbacks.remove(reduce_lr)
    #     steps_per_epoch = max(1, num_train//args.batch_size)
    #     decay_steps = steps_per_epoch * (args.total_epoch - args.init_epoch) #  - args.transfer_epoch
    #     optimizer = get_optimizer(args.optimizer, args.learning_rate, decay_type=args.decay_type, decay_steps=decay_steps)
    #
    # # Unfreeze the whole network for further tuning
    # # NOTE: more GPU memory is required after unfreezing the body
    # print("Unfreeze the whole model to fine-tune.")
    # if args.gpu_num >= 2:
    #     with strategy.scope():
    #         for i in range(len(model.layers)):
    #             model.layers[i].trainable = True
    #         model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
    # else:
    #     for i in range(len(model.layers)):
    #         model.layers[i].trainable = True
    #     model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change







    #### Custom model pipeline  ##############################
    # TODO: update yolo3/data.py _parse_train_data2 returns
    # TODO: update yolo4/models/yolo4_darknet yolo4_body output

    def build_optimizer(optimizer_type, learning_rate, decay_type, num_train, batch_size, total_epoch, init_epoch):
        optimizer = get_optimizer(optimizer_type, learning_rate, decay_type=None)
        if decay_type:
            # rebuild optimizer to apply learning rate decay, only after
            # unfreeze all layers
            # callbacks.remove(reduce_lr)
            steps_per_epoch = max(1, num_train//batch_size)
            decay_steps = steps_per_epoch * (total_epoch - init_epoch) #  - args.transfer_epoch
            optimizer = get_optimizer(optimizer_type, learning_rate, decay_type=decay_type, decay_steps=decay_steps)
        return optimizer

    def create_train_step_fn(strategy, model, loss_fn, optimizer):
        """Creates a distributed training step"""

        def _train_step_fn(inputs):
            inputs, labels = inputs
            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)
                all_losses = loss_fn(labels, outputs)
                losses = {}
                for k, v in all_losses.items():
                    losses[k] = tf.reduce_mean(v)
                per_replica_loss = losses['total_loss'] / strategy.num_replicas_in_sync

            grads = tape.gradient(per_replica_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return losses

        @tf.function
        def train_step(dataset_inputs):
            per_replica_losses = strategy.run(_train_step_fn, args=(dataset_inputs,))
            losses = tf.nest.map_structure(lambda x: strategy.reduce(tf.distribute.ReduceOp.MEAN, x, axis=None),
                                           per_replica_losses)
            return losses

        return train_step

    def train(train_step, train_dist_dataset, initial_epoch, initial_step,
              epochs, steps_per_epoch, checkpoint_manager, optimizer, print_freq):

        # train_summary_writer = SummaryWriter(log_dir, 'train')
        # validation_summary_writer = SummaryWriter(log_dir, 'validation')
        # compression_summary_writer = SummaryWriter(log_dir, 'compression')

        timer = Timer()
        timer.tic()

        logger.info('Training...')
        for epoch in range(initial_epoch, epochs):
            logger.info('Epoch: {}/{}'.format(epoch, epochs))
            # compression_ctrl.scheduler.epoch_step(epoch)

            for step, x in enumerate(train_dist_dataset):
                if epoch == initial_epoch and step < initial_step % steps_per_epoch:
                    continue
                # if step == steps_per_epoch:
                #     save_path = checkpoint_manager.save()
                #     logger.info('Saved checkpoint for epoch={}: {}'.format(epoch, save_path))
                #     break

                # compression_ctrl.scheduler.step()
                train_loss = train_step(x)
                train_metric_result = tf.nest.map_structure(lambda s: s.numpy().astype(float), train_loss)

                if np.isnan(train_metric_result['total_loss']):
                    raise ValueError('total loss is NaN')

                train_metric_result.update({'learning_rate': optimizer.lr(optimizer.iterations).numpy()})

                # train_summary_writer(metrics=train_metric_result, step=optimizer.iterations.numpy())

                if step % print_freq == 0:
                    time = timer.toc(average=False)
                    logger.info('Step: {}/{} Time: {:.3f} sec'.format(step, steps_per_epoch, time))
                    logger.info('Training metric = {}'.format(train_metric_result))
                    timer.tic()

            # test_metric_result = evaluate(test_step, eval_metric, test_dist_dataset, num_test_batches, print_freq)
            # validation_summary_writer(metrics=test_metric_result, step=optimizer.iterations.numpy())
            # eval_metric.reset_states()
            # logger.info('Validation metric = {}'.format(test_metric_result))

            # statistics = compression_ctrl.statistics()
            # print_statistics(statistics)
            # statistics = {'compression/statistics/' + key: value
            #               for key, value in statistics.items()
            #               if isinstance(value, (int, float))}
            # compression_summary_writer(metrics=statistics,
            #                            step=optimizer.iterations.numpy())

        # train_summary_writer.close()
        # validation_summary_writer.close()
        # compression_summary_writer.close()

    # Training parameters
    epochs = args.total_epoch
    steps_per_epoch = train_builder.steps_per_epoch

    with strategy.scope():
        model = get_yolo3_train_model_custom(anchors, num_classes, weights_path=args.weights_path,
                                             label_smoothing=args.label_smoothing, elim_grid_sense=args.elim_grid_sense)
        optimizer = build_optimizer(args.optimizer, args.learning_rate, args.decay_type, num_train, args.batch_size,
                                    args.total_epoch, args.init_epoch)

        loss_fn = build_loss_fn(anchors, num_classes, label_smoothing=args.label_smoothing, elim_grid_sense=args.elim_grid_sense)

        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, config['checkpoint_save_dir'], max_to_keep=None)

        initial_epoch = initial_step = 0

    train_step = create_train_step_fn(strategy, model, loss_fn, optimizer)

    train(train_step, train_dist_dataset, initial_epoch, initial_step,
          epochs, steps_per_epoch, checkpoint_manager, optimizer, config['print_freq'])





    print('!!! Test train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(train_builder.num_examples, test_builder.num_examples, args.batch_size, input_shape))
    # TODO: update yolo3/data.py _parse_train_data2 returns
    # TODO: update yolo4/models/yolo4_darknet yolo4_body output
    model.fit(
        train_dataset,
        steps_per_epoch=max(1, train_builder.num_examples // args.batch_size),
        validation_data=test_dataset,
        validation_steps=max(1, test_builder.num_examples // args.batch_size),
        epochs=args.total_epoch,
        initial_epoch=args.init_epoch,
        workers=10,
        use_multiprocessing=False,
        max_queue_size=10,
        callbacks=callbacks
    )




    # print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(num_train, num_val, args.batch_size, input_shape))
    # #model.fit_generator(train_data_generator,
    # model.fit_generator(data_generator(dataset[:num_train], args.batch_size, input_shape, anchors, num_classes, args.enhance_augment, rescale_interval, multi_anchor_assign=args.multi_anchor_assign),
    #     steps_per_epoch=max(1, num_train//args.batch_size),
    #     #validation_data=val_data_generator,
    #     validation_data=data_generator(dataset[num_train:], args.batch_size, input_shape, anchors, num_classes, multi_anchor_assign=args.multi_anchor_assign),
    #     validation_steps=max(1, num_val//args.batch_size),
    #     epochs=args.total_epoch,
    #     initial_epoch=args.init_epoch,
    #     #verbose=1,
    #     workers=1,
    #     use_multiprocessing=False,
    #     max_queue_size=10,
    #     callbacks=callbacks)

    # Finally store model
    model.save(os.path.join(log_dir, 'trained_final.h5'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model definition options
    parser.add_argument('--model_type', type=str, required=False, default='yolo3_mobilenet_lite',
        help='YOLO model type: yolo3_mobilenet_lite/tiny_yolo3_mobilenet/yolo3_darknet/..., default=%(default)s')
    parser.add_argument('--anchors_path', type=str, required=False, default=os.path.join('configs', 'yolo3_anchors.txt'),
        help='path to anchor definitions, default=%(default)s')
    parser.add_argument('--model_image_size', type=str, required=False, default='416x416',
        help = "Initial model image input size as <height>x<width>, default=%(default)s")
    parser.add_argument('--weights_path', type=str, required=False, default=None,
        help = "Pretrained model/weights file for fine tune")

    # Data options
    parser.add_argument('--annotation_file', type=str, required=False, default='trainval.txt',
        help='train annotation txt file, default=%(default)s')
    parser.add_argument('--val_annotation_file', type=str, required=False, default=None,
        help='val annotation txt file, default=%(default)s')
    parser.add_argument('--val_split', type=float, required=False, default=0.1,
        help = "validation data persentage in dataset if no val dataset provide, default=%(default)s")
    parser.add_argument('--classes_path', type=str, required=False, default=os.path.join('configs', 'voc_classes.txt'),
        help='path to class definitions, default=%(default)s')

    # Training options
    parser.add_argument('--batch_size', type=int, required=False, default=16,
        help = "Batch size for train, default=%(default)s")
    parser.add_argument('--optimizer', type=str, required=False, default='adam', choices=['adam', 'rmsprop', 'sgd'],
        help = "optimizer for training (adam/rmsprop/sgd), default=%(default)s")
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-3,
        help = "Initial learning rate, default=%(default)s")
    parser.add_argument('--decay_type', type=str, required=False, default=None, choices=[None, 'cosine', 'exponential', 'polynomial', 'piecewise_constant'],
        help = "Learning rate decay type, default=%(default)s")
    # parser.add_argument('--transfer_epoch', type=int, required=False, default=20,
    #     help = "Transfer training (from Imagenet) stage epochs, default=%(default)s")
    # parser.add_argument('--freeze_level', type=int,required=False, default=None, choices=[None, 0, 1, 2],
    #     help = "Freeze level of the model in transfer training stage. 0:NA/1:backbone/2:only open prediction layer")
    parser.add_argument('--init_epoch', type=int,required=False, default=0,
        help = "Initial training epochs for fine tune training, default=%(default)s")
    parser.add_argument('--total_epoch', type=int,required=False, default=250,
        help = "Total training epochs, default=%(default)s")
    parser.add_argument('--multiscale', default=False, action="store_true",
        help='Whether to use multiscale training')
    parser.add_argument('--rescale_interval', type=int, required=False, default=10,
        help = "Number of iteration(batches) interval to rescale input size, default=%(default)s")
    parser.add_argument('--enhance_augment', type=str, required=False, default=None, choices=[None, 'mosaic'],
        help = "enhance data augmentation type (None/mosaic), default=%(default)s")
    parser.add_argument('--label_smoothing', type=float, required=False, default=0,
        help = "Label smoothing factor (between 0 and 1) for classification loss, default=%(default)s")
    parser.add_argument('--multi_anchor_assign', default=False, action="store_true",
        help = "Assign multiple anchors to single ground truth")
    parser.add_argument('--elim_grid_sense', default=False, action="store_true",
        help = "Eliminate grid sensitivity")
    parser.add_argument('--data_shuffle', default=False, action="store_true",
        help='Whether to shuffle train/val data for cross-validation')
    parser.add_argument('--gpu_num', type=int, required=False, default=1,
        help='Number of GPU to use, default=%(default)s')
    parser.add_argument('--model_pruning', default=False, action="store_true",
        help='Use model pruning for optimization, only for TF 1.x')

    # Evaluation options
    parser.add_argument('--eval_online', default=False, action="store_true",
        help='Whether to do evaluation on validation dataset during training')
    parser.add_argument('--eval_epoch_interval', type=int, required=False, default=10,
        help = "Number of iteration(epochs) interval to do evaluation, default=%(default)s")
    parser.add_argument('--save_eval_checkpoint', default=False, action="store_true",
        help='Whether to save checkpoint with best evaluation result')

    args = parser.parse_args()
    height, width = args.model_image_size.split('x')
    args.model_image_size = (int(height), int(width))

    main(args)
