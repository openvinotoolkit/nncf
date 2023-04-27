# pylint:skip-file
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
