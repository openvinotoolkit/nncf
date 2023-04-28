import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.keras.saving import saving_utils


def to_frozen_graph(model: tf.keras.Model):
    """
    Returns a frozen graph def for a tf.keras model.
    :param model: The tf.keras model we want to convert.
    :return: The frozen graph def for the provided model.
    """
    func = saving_utils.trace_model_call(model)
    concrete_func = func.get_concrete_function()

    graph_captures = concrete_func.graph._captures  # pylint: disable=protected-access
    captured_inputs = [t_name.name for t_val, t_name in graph_captures.values()]

    input_names = [
        input_tensor.name for input_tensor in concrete_func.inputs if input_tensor.name not in captured_inputs
    ]

    output_names = [
        output_tensor.name for output_tensor in concrete_func.outputs if output_tensor.dtype != tf.dtypes.resource
    ]

    with tf.device("/cpu:0"):
        frozen_func = convert_variables_to_constants_v2(
            concrete_func, lower_control_flow=False, aggressive_inlining=True
        )
        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
        with tf.Graph().as_default():  # pylint: disable=not-context-manager
            tf.import_graph_def(graph_def, name="")
            frozen_graph = tf_optimize_grappler(input_names, output_names, graph_def)

    return frozen_graph


def tf_optimize_grappler(input_names, output_names, graph_def):
    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    config.graph_options.infer_shapes = True
    rewrite_options.optimizers[:] = [
        "pruning",
        "constfold",
        "arithmetic",
        "dependency",
        "function",
    ]

    meta_graph = tf.compat.v1.train.export_meta_graph(graph_def=graph_def)

    fetch_collection = meta_graph_pb2.CollectionDef()
    for t in input_names + output_names:
        fetch_collection.node_list.value.append(t)
    meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

    graph_def = tf_optimizer.OptimizeGraph(config, meta_graph)
    return graph_def
