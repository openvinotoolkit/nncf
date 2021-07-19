"""
 Copyright (c) 2020 Intel Corporation
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

import os
import tensorflow as tf


def get_distribution_strategy(config):
    if config.get('cpu_only', False):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return tf.distribute.OneDeviceStrategy('device:CPU:0')

    gpu_id = config.get('gpu_id', None)
    if gpu_id is not None:
        _gpu_id = str(gpu_id)
        if 'CUDA_VISIBLE_DEVICES' not in os.environ \
                or _gpu_id in os.environ['CUDA_VISIBLE_DEVICES'].split(','):
            os.environ['CUDA_VISIBLE_DEVICES'] = _gpu_id
        else:
            raise RuntimeError('GPU with id = {id} was not found in the specified '
                               'CUDA_VISIBLE_DEVICES environment variable. '
                               'Please do not export the CUDA_VISIBLE_DEVICES environment variable '
                               'or specify GPU with id = {id} in it'.format(id=_gpu_id))

    num_gpus = len(tf.config.list_physical_devices('GPU'))

    if num_gpus > 1:
        return tf.distribute.MirroredStrategy()

    return tf.distribute.get_strategy()
