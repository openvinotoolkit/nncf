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
    - Calculate activation statistics
    - Compare activations between two model variants
    - Register custom comparator metrics and visualizers
    
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
        
        # Register custom metrics
        @NNCFProfiler.comparator("custom_metric")
        def custom_metric(a, b):
            return float(np.abs(a - b).max())
        
        # Compare with custom metrics
        comparison = profiler.compare_activations(
            acts1, acts2, 
            metrics=["mean_diff", "custom_metric"]
        )
        ```
    """
    
    # Class-level registries for comparators, visualizers, and statistics
    COMPARATORS: Dict[str, Any] = {}
    VISUALIZERS: Dict[str, Any] = {}
    STATISTICS: Dict[str, Any] = {}
    
    @classmethod
    def comparator(cls, name: str):
        """
        Decorator to register a custom comparator function for activation comparison.
        
        A comparator function should take two arguments (data1, data2) representing
        activation arrays and return a scalar metric value.
        
        Example:
            @NNCFProfiler.comparator("mean_diff")
            def mean_diff(a: np.ndarray, b: np.ndarray) -> float:
                return float((a - b).mean())
        
        Args:
            name: Name to register the comparator under
        
        Returns:
            Decorator function that registers the comparator
        """
        def decorator(func):
            cls.COMPARATORS[name] = func
            return func
        return decorator
    
    @classmethod
    def visualizer(cls, name: str):
        """
        Decorator to register a custom visualizer function for plotting activations.
        
        A visualizer function should accept activation data and plotting parameters,
        and return a matplotlib figure or display a plot.
        
        Example:
            @NNCFProfiler.visualizer("hist")
            def hist_plot(data: ActivationData, layer: str, **kwargs):
                import matplotlib.pyplot as plt
                # ... plotting code ...
                return plt.gcf()
        
        Args:
            name: Name to register the visualizer under
        
        Returns:
            Decorator function that registers the visualizer
        """
        def decorator(func):
            cls.VISUALIZERS[name] = func
            return func
        return decorator
    
    @classmethod
    def statistic(cls, name: str):
        """
        Decorator to register a custom statistic function for calculating activation statistics.
        
        A statistic function should take a numpy array representing flattened activation values
        and return a scalar statistic value.
        
        Example:
            @NNCFProfiler.statistic("median")
            def median(vals: np.ndarray) -> float:
                return float(np.median(vals))
        
        Args:
            name: Name to register the statistic under
        
        Returns:
            Decorator function that registers the statistic
        """
        def decorator(func):
            cls.STATISTICS[name] = func
            return func
        return decorator

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


    def calculate_stats(
            self,
            data: ActivationData,
            statistics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate custom statistics for collected activations using registered statistic functions.
        
        Args:
            data: Activation data dictionary as returned by collect_activations().
                  Format: {layer_name: {'in': [arrays...], 'out': [arrays...]}}
            statistics: List of statistic names to compute. If None, uses all registered statistics.
                       Statistics must be registered in STATISTICS registry.
        
        Returns:
            pandas DataFrame with columns:
            - name: Layer name
            - type: Activation type ('in' or 'out')
            - <statistic_name>: One column for each requested statistic
        
        Raises:
            ValueError: If any statistic is not found in STATISTICS registry
        
        Example:
            >>> # Register custom statistic
            >>> @NNCFProfiler.statistic("median")
            >>> def median(vals):
            ...     return float(np.median(vals))
            >>> 
            >>> profiler.calculate_stats(acts, statistics=["min", "max", "median"])
        """
        # Use all registered statistics if not specified
        if statistics is None:
            statistics = list(self.STATISTICS.keys())
        
        # Validate all statistics are registered
        for stat in statistics:
            if stat not in self.STATISTICS:
                raise ValueError(
                    f"Unknown statistic '{stat}'. "
                    f"Available: {list(self.STATISTICS.keys())}"
                )
        
        activation_types = ['in', 'out']
        result_data = {'name': [], 'type': []}
        
        # Initialize columns for each statistic
        for stat in statistics:
            result_data[stat] = []
        
        # Calculate statistics for each layer and activation type
        for layer_name in data:
            result_data['name'].extend([layer_name] * len(activation_types))
            result_data['type'].extend(activation_types)
            
            for activation_type in activation_types:
                # Get all activation arrays for this type and flatten them
                activation_arrays = data[layer_name][activation_type]
                flattened_vals = np.concatenate([arr.flatten() for arr in activation_arrays])
                
                # Calculate each statistic using registered functions
                for stat in statistics:
                    stat_value = self.STATISTICS[stat](flattened_vals)
                    result_data[stat].append(stat_value)
        
        return pd.DataFrame(result_data)


    def compare_activations(
        self,
        data1: ActivationData,
        data2: ActivationData,
        metrics: Optional[List[str]] = None,
        statistics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare activations between two model variants using specified metrics and statistics.
        
        Args:
            data1: Activation data from the first model (baseline)
            data2: Activation data from the second model (modified)
            metrics: List of comparator metric names to compute. If None, uses all registered comparators.
                    Metrics must be registered in COMPARATORS registry.
            statistics: List of statistic names to compute for each dataset. If None, uses all registered statistics.
                       Statistics must be registered in STATISTICS registry.
        
        Returns:
            pandas DataFrame containing:
            - name: Layer name
            - type: Activation type ('in' or 'out')
            - <stat>_data1: Statistics for data1 (e.g., 'mean_data1', 'std_data1')
            - <stat>_data2: Statistics for data2 (e.g., 'mean_data2', 'std_data2')
            - <metric>: Comparator metrics (e.g., 'mean_diff', 'relative_diff')
        
        Raises:
            ValueError: If any metric or statistic is not found in respective registries
        
        Example:
            >>> profiler.compare_activations(acts_fp32, acts_int8, 
            ...                  metrics=["mean_diff", "relative_diff"],
            ...                  statistics=["min", "max", "mean", "std"])
        """
        # Verify both datasets have the same layers
        assert set(data1.keys()) == set(data2.keys()), \
            "Activation datasets must contain the same layer names"
        
        # Use all registered comparators if metrics not specified
        if metrics is None:
            metrics = list(self.COMPARATORS.keys())
        
        # Use all registered statistics if not specified
        if statistics is None:
            statistics = list(self.STATISTICS.keys())
        
        # Validate all metrics are registered
        for metric in metrics:
            if metric not in self.COMPARATORS:
                raise ValueError(
                    f"Unknown comparator '{metric}'. "
                    f"Available: {list(self.COMPARATORS.keys())}"
                )
        
        # Validate all statistics are registered
        for stat in statistics:
            if stat not in self.STATISTICS:
                raise ValueError(
                    f"Unknown statistic '{stat}'. "
                    f"Available: {list(self.STATISTICS.keys())}"
                )
        
        # Calculate statistics for data1 and data2 using calculate_stats() method
        stats1_df = self.calculate_stats(data1, statistics=statistics)
        stats2_df = self.calculate_stats(data2, statistics=statistics)
        
        # Rename statistic columns to indicate which dataset they belong to
        for stat in statistics:
            stats1_df.rename(columns={stat: f'{stat}_data1'}, inplace=True)
            stats2_df.rename(columns={stat: f'{stat}_data2'}, inplace=True)
        
        # Merge statistics dataframes
        result = stats1_df.copy()
        for col in stats2_df.columns:
            if col not in ['name', 'type']:
                result[col] = stats2_df[col]
        
        # Calculate comparator metrics for each layer and activation type
        activation_types = ['in', 'out']
        
        # Initialize columns for each metric
        for metric in metrics:
            result[metric] = None
        
        # Calculate metrics row by row
        row_idx = 0
        for layer_name in data1:
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
                
                # Calculate each metric using registered comparators
                for metric in metrics:
                    metric_value = self.COMPARATORS[metric](vals1, vals2)
                    result.at[row_idx, metric] = metric_value
                
                row_idx += 1
        
        return result


    def plot(self, plot_type: str, *args, **kwargs):
        """
        Generate visualizations using registered visualizer functions.
        
        Args:
            plot_type: Name of the visualizer to use (must be registered in VISUALIZERS)
            *args: Positional arguments to pass to the visualizer
            **kwargs: Keyword arguments to pass to the visualizer
        
        Returns:
            Return value from the visualizer function (typically a matplotlib figure)
        
        Raises:
            ValueError: If plot_type is not found in VISUALIZERS registry
        
        Example:
            >>> profiler.plot("hist", acts_fp, acts_int, layer="layer_7")
            >>> profiler.plot("mean_std", acts_fp)
            >>> profiler.plot("metric", cmp, metric="relative_diff")
        """
        if plot_type not in self.VISUALIZERS:
            raise ValueError(
                f"Unknown plot type '{plot_type}'. "
                f"Available: {list(self.VISUALIZERS.keys())}"
            )
        return self.VISUALIZERS[plot_type](*args, **kwargs)


# Built-in comparator functions
@NNCFProfiler.comparator("mean_diff")
def _mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the mean of the difference between two activation arrays."""
    return float((a - b).mean())


@NNCFProfiler.comparator("std_diff")
def _std_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the standard deviation of the difference between two activation arrays."""
    return float((a - b).std())


@NNCFProfiler.comparator("relative_diff")
def _relative_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the mean relative difference between two activation arrays."""
    epsilon = 1e-8
    return float((np.abs(a - b) / (np.abs(a) + epsilon)).mean())


# Built-in statistic functions
@NNCFProfiler.statistic("min")
def _stat_min(vals: np.ndarray) -> float:
    """Calculate the minimum value of activations."""
    return float(vals.min())


@NNCFProfiler.statistic("max")
def _stat_max(vals: np.ndarray) -> float:
    """Calculate the maximum value of activations."""
    return float(vals.max())


@NNCFProfiler.statistic("mean")
def _stat_mean(vals: np.ndarray) -> float:
    """Calculate the mean value of activations."""
    return float(vals.mean())


@NNCFProfiler.statistic("std")
def _stat_std(vals: np.ndarray) -> float:
    """Calculate the standard deviation of activations."""
    return float(vals.std())


@NNCFProfiler.statistic("median")
def _stat_median(vals: np.ndarray) -> float:
    """Calculate the median value of activations."""
    return float(np.median(vals))


@NNCFProfiler.statistic("percentile_95")
def _stat_percentile_95(vals: np.ndarray) -> float:
    """Calculate the 95th percentile of activations."""
    return float(np.percentile(vals, 95))


@NNCFProfiler.statistic("percentile_99")
def _stat_percentile_99(vals: np.ndarray) -> float:
    """Calculate the 99th percentile of activations."""
    return float(np.percentile(vals, 99))


@NNCFProfiler.statistic("abs_mean")
def _stat_abs_mean(vals: np.ndarray) -> float:
    """Calculate the mean of absolute values of activations."""
    return float(np.abs(vals).mean())


@NNCFProfiler.visualizer("compare_detailed")
def _compare_detailed_plot(
    data1: ActivationData,
    data2: ActivationData,
    activation_type: Optional[str] = None,
    bins: int = 100,
    show_histograms: bool = True,
    show_summary: bool = True,
    display_figures: bool = False,
    data1_label: str = 'data1',
    data2_label: str = 'data2',
    **kwargs
):
    """
    Create detailed comparison plots between two activation datasets.
    
    This visualizer creates:
    1. Histogram overlays for each layer (if show_histograms=True)
    2. A summary plot with 4 subplots showing:
       - Mean values for data1 across layers
       - Mean values for data2 across layers  
       - Mean difference (data1 - data2) across layers
       - Relative difference across layers
    
    Args:
        data1: First activation dataset (e.g., FP32 model)
        data2: Second activation dataset (e.g., quantized model)
        activation_type: 'in', 'out', or None. If None, processes both 'in' and 'out' activations
        bins: Number of bins for histograms
        show_histograms: Whether to show individual histogram plots for each layer
        show_summary: Whether to show the summary plot with statistics
        display_figures: Whether to display figures immediately. If False, figures are only returned.
                        Set to False to avoid memory issues with many figures.
        data1_label: Label for data1 in plots (e.g., 'fp16', 'fp32')
        data2_label: Label for data2 in plots (e.g., 'int8', 'fp8')
        **kwargs: Additional arguments for plotting
    
    Returns:
        List of matplotlib figures (histograms + summary plot)
    
    Example:
        >>> # Compare both input and output activations
        >>> figs = profiler.plot("compare_detailed", acts_fp32, acts_int8,
        ...                      data1_label='FP32', data2_label='INT8',
        ...                      display_figures=False)
        >>> # Display or save figures as needed
        >>> for i, fig in enumerate(figs):
        ...     fig.savefig(f'comparison_{i}.png')
        ...     plt.close(fig)
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
    
    # Verify both datasets have the same layers
    assert set(data1.keys()) == set(data2.keys()), \
        "Activation datasets must contain the same layer names"
    
    # Determine which activation types to process
    if activation_type is None:
        activation_types = ['in', 'out']
    else:
        activation_types = [activation_type]
    
    all_figures = []
    summary_figures = []
    layer_names = list(data1.keys())
    
    # Prepare data for summary plots (one per activation type)
    summary_data = {}
    for act_type in activation_types:
        summary_data[act_type] = {
            data1_label: {'mean': [], 'std': []},
            data2_label: {'mean': [], 'std': []},
            'diff': {'mean': [], 'std': [], 'relative': []}
        }
    
    # Loop over layers first, then activation types
    for layer_name in layer_names:
        for act_type in activation_types:
            # Get activation arrays
            data1_arrays = data1[layer_name][act_type]
            data2_arrays = data2[layer_name][act_type]
            
            # Flatten first, then concatenate (consistent with other methods)
            data1_flat = np.concatenate([arr.flatten() for arr in data1_arrays])
            data2_flat = np.concatenate([arr.flatten() for arr in data2_arrays])
            
            # Show histogram for this layer and activation type
            if show_histograms:
                fig, ax = plt.subplots(1, figsize=(10, 6))
                fig.suptitle(f'{layer_name} - {act_type}')
                
                ax.hist(data1_flat, bins=bins, alpha=0.5, label=f"{data1_label}")
                ax.hist(data2_flat, bins=bins, alpha=0.5, label=f"{data2_label}")
                ax.set_xlabel('Activation Value')
                ax.set_ylabel('Frequency')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                stats_text = (
                    f'{data1_label}: min={data1_flat.min():.4f}, max={data1_flat.max():.4f}\n'
                    f'{data2_label}: min={data2_flat.min():.4f}, max={data2_flat.max():.4f}'
                )
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                all_figures.append(fig)
                
                # Close figure if not displaying to save memory
                if not display_figures:
                    plt.close(fig)
            
            # Collect statistics for summary plot
            plot_data = summary_data[act_type]
            plot_data[data1_label]['mean'].append(np.mean(data1_flat))
            plot_data[data2_label]['mean'].append(np.mean(data2_flat))
            plot_data['diff']['mean'].append(np.mean(data1_flat - data2_flat))
            plot_data['diff']['relative'].append(
                np.mean(np.abs(data1_flat - data2_flat) / (np.abs(data1_flat) + 1e-8))
            )
            
            plot_data[data1_label]['std'].append(np.std(data1_flat))
            plot_data[data2_label]['std'].append(np.std(data2_flat))
            plot_data['diff']['std'].append(np.std(data1_flat - data2_flat))
    
    # Create summary plots (one for each activation type)
    if show_summary:
        for act_type in activation_types:
            plot_data = summary_data[act_type]
            
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 14))
            fig.suptitle(f'Activation Comparison Summary - {act_type}', fontsize=14)
            
            x = range(len(layer_names))
            
            # Plot 1: data1 mean and std
            ax1.errorbar(x, plot_data[data1_label]['mean'], plot_data[data1_label]['std'],
                        linestyle='none', marker='^', capsize=5, **kwargs)
            ax1.set_ylabel('Mean ± Std')
            ax1.set_title(f'{data1_label} Activations')
            ax1.set_xticks(x)
            ax1.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: data2 mean and std (fix: removed redundant fmt='o')
            ax2.errorbar(x, plot_data[data2_label]['mean'], plot_data[data2_label]['std'],
                        linestyle='none', marker='^', ecolor='green', capsize=5, **kwargs)
            ax2.set_ylabel('Mean ± Std')
            ax2.set_title(f'{data2_label} Activations')
            ax2.set_xticks(x)
            ax2.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: difference mean and std (fix: removed redundant fmt='o')
            ax3.errorbar(x, plot_data['diff']['mean'], plot_data['diff']['std'],
                        linestyle='none', marker='^', ecolor='red', capsize=5, **kwargs)
            ax3.set_ylabel('Mean ± Std')
            ax3.set_title(f'{data1_label} - {data2_label} Difference')
            ax3.set_xticks(x)
            ax3.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            
            # Plot 4: relative difference
            ax4.plot(x, plot_data['diff']['relative'], color='green', marker='o', **kwargs)
            ax4.set_ylabel('Relative Difference')
            ax4.set_title(f'{data1_label} - {data2_label} Relative Difference')
            ax4.set_xlabel('Layer')
            ax4.set_xticks(x)
            ax4.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            summary_figures.append(fig)

            # Close figure if not displaying to save memory
            if not display_figures:
                plt.close(fig)
        return all_figures, summary_figures
    else:
        return all_figures
