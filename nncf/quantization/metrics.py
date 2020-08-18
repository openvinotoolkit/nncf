import numpy as np
import networkx as nx
from copy import deepcopy
from texttable import Texttable
from collections import deque

from nncf.quantization.layers import SymmetricQuantizer
from nncf.utils import get_all_modules_by_type
from nncf.nncf_network import NNCFNetwork, NNCFGraph
from nncf.dynamic_graph.transform_graph import is_nncf_module
from nncf.quantization.quantizer_propagation import DEFAULT_QUANT_TRAIT_TO_OP_DICT, QuantizationTrait

class BaseMetric:
    def __init__(self):
        pass

    def collect(self):
        pass

    def get_metric_table(self):
        pass


class NetworkQuantizationShareMetric(BaseMetric):
    """
    This is a metric representing the share of the model that has been quantized.
    It includes the calculation of the following numbers:
    - Percentage of symmetric/asymmetric/per-channel/per-tensor weight quantizers relative
      to the number of placed weight quantizers
    - Percentage of symmetric/asymmetric/per-channel/per-tensor non weight quantizers relative
      to the number of placed non weight quantizers
    - Percentage of weight quantizers and non weight quantizers for each precision relative
      to the number potential* quantizers / placed quantizers
    Bitwidth distribution data is also collected.
    
    * The maximum possible number of potential quantizers depends on the presence of ignored
    scopes and the mode of quantizer setup that is used at the time of collecting the metric.
    """
    NAME_STR = 'NetworkQuantizationShare'

    WEIGHTS_RATIO_STR = ' WQs / All placed WQs' # WQ - weight quantizer
    ACTIVATIONS_RATIO_STR = ' AQs / All placed AQs' # AQ - activation quantizer
    TOTAL_RATIO_STR = ' Qs (out of total placed)'

    PARAMS_STR = 'Quantizer parameter'
    SYMMETRIC_STR = 'Symmetric'
    ASYMMETRIC_STR = 'Asymmetric'
    PER_CHANNEL_STR = 'Per-channel'
    SIGNED_STR = 'Signed'
    PER_TENSOR_STR = 'Per-tensor'
    UNSIGNED_STR = 'Unsigned'
    SHARE_WEIGHT_QUANTIZERS_STR = 'Placed WQs / Potential WQs'
    SHARE_ACTIVATION_QUANTIZERS_STR = 'Placed AQs / Potential AQs'

    def __init__(self, compressed_model, weights_quantizers, non_weights_quantizers, quantizer_setup_type):
        super().__init__()
        self._compressed_model = compressed_model
        self._quantizer_setup_type = quantizer_setup_type # type: QuantizerSetupType
        self.non_weights_quantizers = {k: v.quantizer_module_ref for k, v in non_weights_quantizers.items()}
        self.weights_quantizers = weights_quantizers
        self._all_quantizations = {**self.weights_quantizers, **self.non_weights_quantizers}
        self.header = [self.PARAMS_STR,self.WEIGHTS_RATIO_STR, self.ACTIVATIONS_RATIO_STR,  self.TOTAL_RATIO_STR]
        self.params = {self.PER_CHANNEL_STR, self.PER_TENSOR_STR, self.UNSIGNED_STR, self.SIGNED_STR,
                       self.SYMMETRIC_STR, self.ASYMMETRIC_STR}
        self.params_bits_stat = set()
        self.num_potential_quantized_weights = len(compressed_model.get_nncf_modules())
        self.num_potential_quantized_activations = self._get_num_potential_quantized_activations()
        self.num_placed_weight_quantizers = len(self.weights_quantizers)
        self.num_placed_activation_quantizers = len(self.non_weights_quantizers)
        self.num_all_potential_quantizer = self.num_potential_quantized_weights + self.num_potential_quantized_activations
        self.stat = {}
        self._ratio = {
            self.WEIGHTS_RATIO_STR: len(self.weights_quantizers),
            self.ACTIVATIONS_RATIO_STR: len(self.non_weights_quantizers),
            self.TOTAL_RATIO_STR: len(self._all_quantizations)}

    def _get_num_potential_quantized_activations(self):
        from nncf.quantization.algo import QuantizerSetupType

        if self._quantizer_setup_type == QuantizerSetupType.PATTERN_BASED:
            from nncf.quantization.algo import QuantizationBuilder
            default_pattern = QuantizationBuilder._make_default_quantizable_subgraph_pattern()
            return len(self._compressed_model.get_post_pattern_insertion_points(default_pattern))
        else:
            from nncf.quantization.algo import QuantizerPropagationSolver
            insertion_point_graph = self._compressed_model.get_insertion_point_graph()
            prop_graph_solver = QuantizerPropagationSolver()
            insertion_data = prop_graph_solver.run_on_ip_graph(insertion_point_graph)
            return len(insertion_data)

    def collect(self):
        for quantizer in self._all_quantizations.values():
            self.params_bits_stat.add(quantizer.num_bits)

        for h in self.header:
            self.stat[h] = {}
            for p in self.params:
                self.stat[h][p] = 0
            for p in self.params_bits_stat:
                self.stat[h][p] = 0

        for quantizer in self._all_quantizations.values():  # type: BaseQuantizer
            num_bits = quantizer.num_bits
            self.stat[self.TOTAL_RATIO_STR][num_bits] += 1
            type_ = self.WEIGHTS_RATIO_STR if quantizer.is_weights else self.ACTIVATIONS_RATIO_STR
            self.stat[type_][num_bits] += 1
            if quantizer.per_channel:
                self.stat[type_][self.PER_CHANNEL_STR] += 1
            else:
                self.stat[type_][self.PER_TENSOR_STR] += 1
            if quantizer.signed:
                self.stat[type_][self.SIGNED_STR] += 1
            else:
                self.stat[type_][self.UNSIGNED_STR] += 1
            if isinstance(quantizer, SymmetricQuantizer):
                self.stat[type_][self.SYMMETRIC_STR] += 1
            else:
                self.stat[type_][self.ASYMMETRIC_STR] += 1

    def _get_copy_statistics(self):
        statistics = deepcopy(self.stat)
        for h in self.header[1:]:
            for key, value in statistics[h].items():
                try:
                    statistics[h][key] /= self._ratio[h]
                    statistics[h][key] *= 100
                except ZeroDivisionError:
                    statistics[h][key]
        return statistics

    def get_metric_table(self):
        table_with_bits_stats = Texttable()
        table_with_other_stats = Texttable() 
        data = [['Metric type', 'Value']]
        for h in (self.WEIGHTS_RATIO_STR, self.ACTIVATIONS_RATIO_STR):
            for p in self.params:   
                try:
                    row = [ '{} '.format(p) + str(h) , '{:.2f} % ({} / {}) '.format(\
                        self.stat[h][p] / self._ratio[h] * 100, self.stat[h][p], self._ratio[h])]
                except ZeroDivisionError:    
                    row = [ '{} '.format(p) + h , 0 ]
                data.append(row)
        try:
            row = [self.SHARE_WEIGHT_QUANTIZERS_STR, '{:.2f} % ({} / {}) '.format(\
                   self.num_placed_weight_quantizers / self.num_potential_quantized_weights * 100 ,
                   self.num_placed_weight_quantizers, self.num_potential_quantized_weights)]
        except ZeroDivisionError:
            row = [self.SHARE_WEIGHT_QUANTIZERS_STR, '{} % '.format(0)]

        data.append(row)
        try:
            row = [self.SHARE_ACTIVATION_QUANTIZERS_STR, '{:.2f} % ({} / {}) '.format(\
                   self.num_placed_activation_quantizers / self.num_potential_quantized_activations * 100 ,
                   self.num_placed_activation_quantizers, self.num_potential_quantized_activations)]
        except ZeroDivisionError:
            row = [self.SHARE_ACTIVATION_QUANTIZERS_STR, '{} % '.format(0)]
        data.append(row)

        table_with_other_stats.add_rows(data)

        data = [['Num bits (N)' , 'N-bits WQs / Placed WQs', 'N-bits AQs / Placed AQs', 'N-bits Qs / Placed Qs']]
        for p in self.params_bits_stat:
            row = [p]
            for h in (self.WEIGHTS_RATIO_STR, self.ACTIVATIONS_RATIO_STR, self.TOTAL_RATIO_STR):
                try:        
                    row.append('{:.2f} % ({} / {}) '.format(\
                        self.stat[h][p] / self._ratio[h] * 100, self.stat[h][p], self._ratio[h]))
                except ZeroDivisionError:    
                    row.append(0)
            data.append(row)
        table_with_bits_stats.add_rows(data)

        retval = {
                  "Share quantization statistics:" : table_with_other_stats,
                  "Bitwidth distribution:" : table_with_bits_stats
        }
        return retval

    def get_bits_stat(self):
        table = Texttable()
        data = [['Num bits (N)' , 'N-bits WQs / Placed Qs', 'N-bits AQs / Placed Qs', 'N-bits Qs / Placed Qs']]
        for p in self.params_bits_stat:
            row = [p]
            for h in (self.WEIGHTS_RATIO_STR, self.ACTIVATIONS_RATIO_STR, self.TOTAL_RATIO_STR):
                try:        
                    row.append(self.stat[h][p] / self._ratio[self.TOTAL_RATIO_STR] * 100)
                except ZeroDivisionError:    
                    row.append(0)
            data.append(row)
        table.add_rows(data)
        return table
