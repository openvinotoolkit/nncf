#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrain the YOLO model for your own dataset.
"""
import os, time, random, argparse
import numpy as np
import tensorflow.keras.backend as K
# from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TerminateOnNaN, LambdaCallback
# from tensorflow_model_optimization.sparsity import keras as sparsity


from yolo3.model import get_yolo3_train_model
from yolo3.data import yolo3_data_generator_wrapper, Yolo3DataGenerator
from common.utils import get_classes, get_anchors, get_dataset, optimize_tf_gpu
from common.model_utils import get_optimizer
# from common.callbacks import EvalCallBack, DatasetShuffleCallBack

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


    # elif args.model_type.startswith('yolo3_') or args.model_type.startswith('yolo4_'):
    #if num_anchors == 9:
    # YOLOv3 & v4 entrance, use 9 anchors
    get_train_model = get_yolo3_train_model
    data_generator = yolo3_data_generator_wrapper

    # tf.keras.Sequence style data generator
    #train_data_generator = Yolo3DataGenerator(dataset[:num_train], args.batch_size, input_shape, anchors, num_classes, args.enhance_augment, rescale_interval, args.multi_anchor_assign)
    #val_data_generator = Yolo3DataGenerator(dataset[num_train:], args.batch_size, input_shape, anchors, num_classes, multi_anchor_assign=args.multi_anchor_assign)

    # tiny_version = False


    # # prepare online evaluation callback
    # if args.eval_online:
    #     eval_callback = EvalCallBack(args.model_type, dataset[num_train:], anchors, class_names, args.model_image_size, args.model_pruning, log_dir, eval_epoch_interval=args.eval_epoch_interval, save_eval_checkpoint=args.save_eval_checkpoint, elim_grid_sense=args.elim_grid_sense)
    #     callbacks.append(eval_callback)
    #
    # # prepare train/val data shuffle callback
    # if args.data_shuffle:
    #     shuffle_callback = DatasetShuffleCallBack(dataset)
    #     callbacks.append(shuffle_callback)

    # prepare model pruning config
    pruning_end_step = np.ceil(1.0 * num_train / args.batch_size).astype(np.int32) * args.total_epoch
    # if args.model_pruning:
    #     pruning_callbacks = [sparsity.UpdatePruningStep(), sparsity.PruningSummaries(log_dir=log_dir, profile_batch=0)]
    #     callbacks = callbacks + pruning_callbacks

    # prepare optimizer
    optimizer = get_optimizer(args.optimizer, args.learning_rate, decay_type=None)

    # support multi-gpu training
    if args.gpu_num >= 2:
        # devices_list=["/gpu:0", "/gpu:1"]
        devices_list=["/gpu:{}".format(n) for n in range(args.gpu_num)]
        strategy = tf.distribute.MirroredStrategy(devices=devices_list)
        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            # get multi-gpu train model
            model = get_train_model(args.model_type, anchors, num_classes, weights_path=args.weights_path, optimizer=optimizer, label_smoothing=args.label_smoothing, elim_grid_sense=args.elim_grid_sense, model_pruning=args.model_pruning, pruning_end_step=pruning_end_step)

    else:
        # get normal train model
        model = get_train_model(args.model_type, anchors, num_classes, weights_path=args.weights_path, optimizer=optimizer, label_smoothing=args.label_smoothing, elim_grid_sense=args.elim_grid_sense, model_pruning=args.model_pruning, pruning_end_step=pruning_end_step)

    model.summary()

    # # Transfer training some epochs with frozen layers first if needed, to get a stable loss.
    # initial_epoch = args.init_epoch
    # epochs = initial_epoch + args.transfer_epoch
    # print("Transfer training stage")
    # print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(num_train, num_val, args.batch_size, input_shape))
    # #model.fit_generator(train_data_generator,
    # model.fit_generator(data_generator(dataset[:num_train], args.batch_size, input_shape, anchors, num_classes, args.enhance_augment, rescale_interval, multi_anchor_assign=args.multi_anchor_assign),
    #         steps_per_epoch=max(1, num_train//args.batch_size),
    #         #validation_data=val_data_generator,
    #         validation_data=data_generator(dataset[num_train:], args.batch_size, input_shape, anchors, num_classes, multi_anchor_assign=args.multi_anchor_assign),
    #         validation_steps=max(1, num_val//args.batch_size),
    #         epochs=epochs,
    #         initial_epoch=initial_epoch,
    #         #verbose=1,
    #         workers=1,
    #         use_multiprocessing=False,
    #         max_queue_size=10,
    #         callbacks=callbacks)
    #
    # # Wait 2 seconds for next stage
    # time.sleep(2)

    if args.decay_type:
        # rebuild optimizer to apply learning rate decay, only after
        # unfreeze all layers
        callbacks.remove(reduce_lr)
        steps_per_epoch = max(1, num_train//args.batch_size)
        decay_steps = steps_per_epoch * (args.total_epoch - args.init_epoch) #  - args.transfer_epoch
        optimizer = get_optimizer(args.optimizer, args.learning_rate, decay_type=args.decay_type, decay_steps=decay_steps)

    # Unfreeze the whole network for further tuning
    # NOTE: more GPU memory is required after unfreezing the body
    print("Unfreeze the whole model to fine-tune.")
    if args.gpu_num >= 2:
        with strategy.scope():
            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change

    else:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change

    # initial_epoch = args.init_epoch
    # epochs = initial_epoch

    print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(num_train, num_val, args.batch_size, input_shape))
    #model.fit_generator(train_data_generator,
    model.fit_generator(data_generator(dataset[:num_train], args.batch_size, input_shape, anchors, num_classes, args.enhance_augment, rescale_interval, multi_anchor_assign=args.multi_anchor_assign),
        steps_per_epoch=max(1, num_train//args.batch_size),
        #validation_data=val_data_generator,
        validation_data=data_generator(dataset[num_train:], args.batch_size, input_shape, anchors, num_classes, multi_anchor_assign=args.multi_anchor_assign),
        validation_steps=max(1, num_val//args.batch_size),
        epochs=args.total_epoch,
        initial_epoch=args.init_epoch,
        #verbose=1,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
        callbacks=callbacks)

    # Finally store model
    if args.model_pruning:
        model = sparsity.strip_pruning(model)
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
