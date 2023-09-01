from nncf.quantization.algorithms.post_training.backend import PostTrainingBackend


class PTPostTrainingBackend(PostTrainingBackend):
    def set_subgraph(self, subgraph_model, if_op, if_op_subgraph_port_id):
        pass

    def dump_model(self, model, dir, if_op, if_op_subgraph_port_id):
        pass

    def is_single_model(self, model):
        return True

    def add_results(self, model, node):
        pass

    def collect_dataset(self, model, ifnode, calibration_dataset, subset_size):
        pass

    def make_tasks(self, model, calibration_dataset, subset_size):
        pass
