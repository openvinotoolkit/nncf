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

"""
NNCF Profiler Tool

A tool for collecting activation statistics from OpenVINO models using NNCF infrastructure.
This profiler can collect raw activations at specific layers matching regex patterns.
"""

import re
import sys
from typing import Any, Dict, List, Optional, Pattern, Union, Tuple

import numpy as np
import openvino.runtime as ov
import pandas as pd

from nncf.common.tensor_statistics.statistic_point import StatisticPoint, StatisticPointsContainer
from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator, RawReducer, TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import RawTensorStatistic
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.graph.transformations.commands import OVTargetPoint, TargetType
from nncf.openvino.statistics.aggregator import OVStatisticsAggregator
from nncf.openvino.statistics.collectors import get_raw_stat_collector

# Type aliases for better readability
ActivationData = Dict[str, Dict[str, List[np.ndarray]]]
StatisticPointsList = List[Any]

class NNCFProfiler:
    """
    A profiler for collecting activation statistics from OpenVINO models.
    
    This class provides functionality to:
    - Collect raw activations at input and output of specific layers
    - Filter layers using regex patterns
    - Aggregate statistics across multiple samples
    - Calculate activation statistics (min, max, mean, std)
    - Compare activations between two model variants
    
    Attributes:
        dataset: Dataset object for collecting statistics (typically nncf.Dataset)
        pattern: Regex pattern to match layer names for activation collection
        num_samples: Number of samples to collect from the dataset
    
    Example usage:
        ```python
        import openvino.runtime as ov
        from nncf import Dataset
        
        model = ov.Core().read_model("model.xml")
        dataset = Dataset(data_source, transform_fn)
        
        profiler = NNCFProfiler(
            pattern=r'__module.model.layers.\d+.self_attn',
            dataset=dataset,
            num_samples=100
        )
        
        # Collect activations from specific layers
        activations = profiler.collect_activations(model)
        
        # Access collected data
        for layer_name, data in activations.items():
            input_activations = data['in']   # List[np.ndarray]
            output_activations = data['out']  # List[np.ndarray]
        ```
    """

    def __init__(
        self, 
        pattern: Union[str, Pattern[str]], 
        dataset: Any, 
        num_samples: int
    ) -> None:
        """
        Initialize the NNCF Profiler.
        
        Args:
            pattern: Regex pattern (string or compiled Pattern object) to match layer names.
                     Examples: r'self_attn', r'__module.model.layers.\d+.mlp'
            dataset: Dataset object for collecting statistics. Should be compatible with
                     NNCF's OVStatisticsAggregator (typically nncf.Dataset)
            num_samples: Number of samples to collect from the dataset for profiling.
                        Larger values provide more accurate statistics but consume more memory
        """
        self.dataset: Any = dataset
        self.pattern: Union[str, Pattern[str]] = pattern
        self.num_samples: int = num_samples
    

    def _get_statistic_points(
        self, 
        model: ov.Model, 
        graph: Any, 
        nodes: List[Any], 
        subset_size: int
    ) -> StatisticPointsContainer:
        """
        Create statistic points for collecting activations at input and output of specified nodes.
        
        This is an internal helper method that configures collection points for both pre-layer
        (input) and post-layer (output) activations for each target node.
        
        Args:
            model: OpenVINO model being profiled
            graph: NNCF graph representation of the model
            nodes: List of NNCF graph nodes to collect statistics from
            subset_size: Number of samples to collect for each statistic point
        
        Returns:
            StatisticPointsContainer with configured collection points for all target nodes
        """
        statistic_container = StatisticPointsContainer()
        OUTPUT_PORT_OF_NODE = 0
        INPUT_PORT_OF_NODE = 0

        # Collection of statistics after/before layers.
        for node in nodes:
            node_name = node.node_name
            channel_axis = node.metatype.output_channel_axis
            if channel_axis is None:
                channel_axis = -1

            # For layers with weights, there is only one output port - 0.
            statistic_point_out = OVTargetPoint(
                TargetType.POST_LAYER_OPERATION, 
                node_name, 
                port_id=OUTPUT_PORT_OF_NODE
            )
            stat_collector_out = get_raw_stat_collector(num_samples=subset_size)
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=statistic_point_out, 
                    tensor_collector=stat_collector_out, 
                    algorithm="collect"
                )
            )
            

            # For layers with weights, there is only one output port - 0.
            statistic_point_in = OVTargetPoint(
                TargetType.PRE_LAYER_OPERATION, 
                node_name, 
                port_id=INPUT_PORT_OF_NODE
            )
            stat_collector_in = get_raw_stat_collector(num_samples=subset_size)
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=statistic_point_in, 
                    tensor_collector=stat_collector_in, 
                    algorithm="collect"
                )
            )

        return statistic_container

    def collect_activations(
        self, 
        model: ov.Model, 
        dataset: Optional[Any] = None, 
        pattern: Optional[Union[str, Pattern[str]]] = None, 
        num_samples: Optional[int] = None
    ) -> ActivationData:
        """
        Collect activation statistics from layers matching the specified pattern.
        
        This method profiles the model by running inference on the dataset and collecting
        raw activation values at both the input and output of matching layers.
        
        Args:
            model: OpenVINO model to profile
            dataset: Optional dataset to use for collection. If None, uses the instance's dataset.
                    Should be compatible with NNCF's OVStatisticsAggregator
            pattern: Optional regex pattern to match layer names. If None, uses the instance's pattern.
                    Examples: r'self_attn', r'__module.model.layers.\d+.mlp'
            num_samples: Optional number of samples to collect. If None, uses the instance's num_samples
        
        Returns:
            Dictionary mapping layer names to their collected activations:
            {
                'layer_name_1': {
                    'in': [array1, array2, ...],   # List of input activation arrays
                    'out': [array1, array2, ...]   # List of output activation arrays
                },
                'layer_name_2': {...},
                ...
            }
            Each array is a numpy array containing activation values for one sample.
        
        Raises:
            ValueError: If no layers match the specified pattern
        """
        # Use provided parameters or fall back to instance defaults
        dataset = dataset if dataset is not None else self.dataset
        pattern = pattern if pattern is not None else self.pattern
        num_samples = num_samples if num_samples is not None else self.num_samples
        
        # Compile pattern if it's a string
        regexp: Pattern[str] = re.compile(pattern) if isinstance(pattern, str) else pattern

        # Build graph and create statistics aggregator
        graph = GraphConverter.create_nncf_graph(model)
        statistics_aggregator = OVStatisticsAggregator(dataset)
        
        # Find target nodes matching the pattern
        node_keys = graph.get_all_node_keys()
        target_names = [key for key in node_keys if regexp.search(key)]
        
        if not target_names:
            raise ValueError(f"No layers found matching pattern: {pattern}")
        
        target_ops = [graph.get_node_by_key(name) for name in target_names]

        # Register statistic collection points and collect statistics
        statistic_points = self._get_statistic_points(model, graph, target_ops, num_samples)
        statistics_aggregator.register_statistic_points(statistic_points)
        statistics_aggregator.collect_statistics(model, graph)

        # Extract and convert collected statistics to numpy arrays
        result: ActivationData = {}
        for layer_name, statistic_points_list in statistics_aggregator.statistic_points.items():
            # Extract input activations (index 1 in statistic_points_list)
            in_container = list(
                statistic_points_list[1]
                .algorithm_to_tensor_collectors["collect"][0]
                .aggregators.values()
            )[0]._container
            in_vals = [np.array(elem.data) for elem in in_container]
            
            # Extract output activations (index 0 in statistic_points_list)
            out_container = list(
                statistic_points_list[0]
                .algorithm_to_tensor_collectors["collect"][0]
                .aggregators.values()
            )[0]._container
            out_vals = [np.array(elem.data) for elem in out_container]
            
            result[layer_name] = {
                'in': in_vals,
                'out': out_vals
            }
        
        return result

    def calculate_activation_stats(self, data: ActivationData) -> pd.DataFrame:
        """
        Calculate statistical measures (min, max, mean, std) for collected activations.
        
        This method processes the collected activation data and computes summary statistics
        for both input and output activations of each layer.
        
        Args:
            data: Activation data dictionary as returned by collect_activations().
                  Format: {layer_name: {'in': [arrays...], 'out': [arrays...]}}
        
        Returns:
            pandas DataFrame with columns:
            - name: Layer name
            - type: Activation type ('in' or 'out')
            - min: Minimum activation value
            - max: Maximum activation value
            - mean: Mean activation value
            - std: Standard deviation of activation values
            
            Each layer will have 2 rows (one for 'in', one for 'out')
        """
        activation_types = ['in', 'out']
        stats_dict = {
            'name': [], 
            'type': [], 
            'min': [], 
            'max': [], 
            'mean': [], 
            'std': []
        }
        
        for layer_name in data:
            # Add layer name for both input and output
            stats_dict['name'].extend([layer_name] * len(activation_types))
            stats_dict['type'].extend(activation_types)
            
            # Calculate statistics for each activation type
            for activation_type in activation_types:
                # Get all activation arrays for this type and flatten them
                activation_arrays = data[layer_name][activation_type]  # List[np.ndarray]
                flattened_vals = np.concatenate([arr.flatten() for arr in activation_arrays])
                
                # Compute statistics
                stats_dict['min'].append(float(flattened_vals.min()))
                stats_dict['max'].append(float(flattened_vals.max()))
                stats_dict['mean'].append(float(flattened_vals.mean()))
                stats_dict['std'].append(float(flattened_vals.std()))
        
        return pd.DataFrame(stats_dict)

    def compare_activations(
        self, 
        data1: ActivationData, 
        data2: ActivationData
    ) -> pd.DataFrame:
        """
        Compare activations between two model variants (e.g., before/after quantization).
        
        This method computes statistics for both datasets and calculates differences to
        help analyze the impact of model modifications on activation distributions.
        
        Args:
            data1: Activation data from the first model (baseline), as returned by collect_activations()
            data2: Activation data from the second model (modified), as returned by collect_activations()
        
        Returns:
            pandas DataFrame containing:
            - name: Layer name
            - type: Activation type ('in' or 'out')
            - min_before, max_before, mean_before, std_before: Statistics from data1
            - min_after, max_after, mean_after, std_after: Statistics from data2
            - mean_diff: Mean of the difference (data1 - data2)
            - std_diff: Standard deviation of the difference
            - relative_diff: Mean relative difference |data1 - data2| / (|data1| + epsilon)
        
        Raises:
            AssertionError: If the two datasets don't have matching layer names or dimensions
        
        Example:
            >>> profiler = NNCFProfiler(pattern=r'attn', dataset=ds, num_samples=100)
            >>> acts_fp32 = profiler.collect_activations(model_fp32)
            >>> acts_int8 = profiler.collect_activations(model_int8)
            >>> comparison = profiler.compare_activations(acts_fp32, acts_int8)
            >>> print(comparison[['name', 'type', 'relative_diff']])
        """
        # Verify both datasets have the same layers
        assert set(data1.keys()) == set(data2.keys()), \
            "Activation datasets must contain the same layer names"
        
        activation_types = ['in', 'out']
        
        # Calculate statistics for both datasets
        df1 = self.calculate_activation_stats(data1)
        df1.columns = [
            col if col in ['name', 'type'] else f'{col}_before' 
            for col in df1.columns
        ]
        
        df2 = self.calculate_activation_stats(data2)
        df2.columns = [
            col if col in ['name', 'type'] else f'{col}_after' 
            for col in df2.columns
        ]
        
        # Merge the two dataframes
        result = df1.copy()
        for col in df2.columns:
            if col not in ['name', 'type']:
                result[col] = df2[col]
        
        # Calculate difference statistics
        diff_stats = {
            'name': [], 
            'type': [], 
            'mean_diff': [], 
            'std_diff': [], 
            'relative_diff': []
        }
        
        for layer_name in data1:
            diff_stats['name'].extend([layer_name] * len(activation_types))
            diff_stats['type'].extend(activation_types)
            
            for activation_type in activation_types:
                vals1_list = data1[layer_name][activation_type]
                vals2_list = data2[layer_name][activation_type]
                
                # Verify dimensions match
                assert len(vals1_list) == len(vals2_list), \
                    f"Number of samples mismatch for {layer_name} {activation_type}"
                
                for i in range(len(vals1_list)):
                    assert vals1_list[i].shape == vals2_list[i].shape, \
                        f"Shape mismatch for {layer_name} {activation_type} sample {i}"
                
                # Flatten and concatenate all samples
                vals1 = np.concatenate([arr.flatten() for arr in vals1_list])
                vals2 = np.concatenate([arr.flatten() for arr in vals2_list])
                
                # Calculate difference metrics
                diff = vals1 - vals2
                epsilon = 1e-8  # Small constant to avoid division by zero
                
                diff_stats['mean_diff'].append(float(diff.mean()))
                diff_stats['std_diff'].append(float(diff.std()))
                diff_stats['relative_diff'].append(
                    float((np.abs(diff) / (np.abs(vals1) + epsilon)).mean())
                )
        
        # Add difference statistics to result dataframe
        for col in diff_stats:
            if col not in ['name', 'type']:
                result[col] = diff_stats[col]
        
        return result
