
import tensorflow as tf


class NNCFNetwork(tf.keras.Model):
    def __init__(self, target_model):
        super().__init__(target_model)
        self.nncf_wrapped_model = target_model
        self.kd_original_model = None
        self.kd_outputs_storage = None
        self.kd_loss = tf.keras.losses.MeanSquaredError()

    def get_nncf_wrapped_model_config(self):
        return self.nncf_wrapped_model.get_config()

    def get_config(self):
        return self.nncf_wrapped_model.get_config()

    def compile(self, *args, **kwargs):
        super(NNCFNetwork, self).compile(*args, **kwargs)
        self.nncf_wrapped_model.compile(*args, **kwargs)

    def call(self, inputs, training=None, mask=None):
        compressed_output = self.nncf_wrapped_model(inputs)
        if self.kd_original_model is not None:
            original_model_outputs = self.kd_original_model(inputs)
            self.kd_outputs_storage = self.kd_loss(original_model_outputs, compressed_output)
        return compressed_output

    def summary(self, *args, **kwargs):
        self.nncf_wrapped_model.summary(*args, **kwargs)

    def enable_knowledge_distillation(self, kd_original_model):
        self.kd_original_model = kd_original_model

    def get_knowledge_distillation_loss_value(self):
        return self.kd_outputs_storage
