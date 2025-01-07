# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import shutil
from pathlib import Path

import tensorflow as tf
from tensorflow.python.platform import gfile

graph_paths = [
    "../tests/tensorflow/data/reference_graphs/quantized/symmetric/per_tensor/retinanet_quantize_outputs.old.pb",
    "../tests/tensorflow/data/reference_graphs/quantized/symmetric/per_tensor/retinanet_quantize_outputs.pb",
]

graphs = {}
tb_log_path = Path("/tmp/tb")  # nosec
if tb_log_path.exists():
    shutil.rmtree(tb_log_path)
    tb_log_path.mkdir()
for pb_path in graph_paths:
    with tf.compat.v1.Session() as sess:
        model_filename = pb_path
        with gfile.FastGFile(model_filename, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
with tf.compat.v1.Graph().as_default():
    train_writer = tf.compat.v1.summary.FileWriter(str(tb_log_path))
    train_writer.add_graph(sess.graph)
    train_writer.close()
