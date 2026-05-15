# Copyright (c) 2026 Intel Corporation
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
from collections import defaultdict
from re import Pattern
from typing import Any

import numpy as np
import openvino as ov
import pandas as pd

from nncf.common.tensor_statistics.builders import get_raw_stat_collector
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.graph.transformations.commands import TargetType
from nncf.openvino.statistics.aggregator import OVStatisticsAggregator

ActivationData = dict[str, dict[str, list[np.ndarray]]]
StatisticPointsList = list[Any]


class NNCFProfiler:
    """
    A profiler for collecting activation statistics from OpenVINO models.

    Core features:
        - Collect input and output activations for selected layers
        - Select layers using regular expression patterns
        - Aggregate activations across multiple dataset samples
        - Compute common activation statistics (mean, std, min/max, etc.)
        - Compare activations between two model variants
        - Register custom comparison metrics and visualization hooks

    :param dataset: Dataset used to run inference and collect activations.
    :param pattern: Regex pattern to match layer names for activation collection
    :param num_samples: Number of samples to collect from the dataset

    Example
    -------
    ```python
    import openvino as ov
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
    COMPARATORS: dict[str, Any] = {}
    VISUALIZERS: dict[str, Any] = {}
    STATISTICS: dict[str, Any] = {}

    def __init__(self, pattern: str | Pattern[str], dataset: Any, num_samples: int) -> None:
        """
        Initialize the NNCF Profiler.

        :param pattern: Regex pattern (string or compiled Pattern object) to match layer names.
            Examples: r'self_attn', r'__module.model.layers.\d+.mlp'
        :param dataset: Dataset object for collecting statistics. Should be compatible with
            NNCF's OVStatisticsAggregator (typically nncf.Dataset).
        :param num_samples: Number of samples to collect from the dataset for profiling.
            Larger values provide more accurate statistics but consume more memory.
        """
        self.dataset: Any = dataset
        self.pattern: str | Pattern[str] = pattern
        self.num_samples: int = num_samples

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

        :param name: Name to register the comparator under
        :return: Decorator function that registers the comparator
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

        :param name: Name to register the visualizer under
        :return: Decorator function that registers the visualizer
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

        :param name: Name to register the statistic under
        :return: Decorator function that registers the statistic
        """

        def decorator(func):
            cls.STATISTICS[name] = func
            return func

        return decorator

    def _get_statistic_points(
        self, model: ov.Model, graph: Any, nodes: list[Any], subset_size: int
    ) -> StatisticPointsContainer:
        """
        Create statistic points for collecting activations at input and output of specified nodes.

        This is an internal helper method that configures collection points for both pre-layer
        (input) and post-layer (output) activations for each target node.

        :param model: OpenVINO model being profiled
        :param graph: NNCF graph representation of the model
        :param nodes: List of NNCF graph nodes to collect statistics from
        :param subset_size: Number of samples to collect for each statistic point
        :return: StatisticPointsContainer with configured collection points for all target nodes
        """
        statistic_container = StatisticPointsContainer()
        output_port_of_node = 0
        input_port_of_node = 0

        # Collection of statistics after/before layers.
        for node in nodes:
            node_name = node.node_name
            channel_axis = node.metatype.output_channel_axis
            if channel_axis is None:
                channel_axis = -1

            # For layers with weights, there is only one output port - 0.
            statistic_point_out = OVTargetPoint(TargetType.POST_LAYER_OPERATION, node_name, port_id=output_port_of_node)
            stat_collector_out = get_raw_stat_collector(num_samples=subset_size)
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=statistic_point_out, tensor_collector=stat_collector_out, algorithm="collect"
                )
            )

            # For layers with weights, there is only one input port - 0.
            statistic_point_in = OVTargetPoint(TargetType.PRE_LAYER_OPERATION, node_name, port_id=input_port_of_node)
            stat_collector_in = get_raw_stat_collector(num_samples=subset_size)
            statistic_container.add_statistic_point(
                StatisticPoint(target_point=statistic_point_in, tensor_collector=stat_collector_in, algorithm="collect")
            )

        return statistic_container

    def collect_activations(
        self,
        model: ov.Model,
        dataset: Any | None = None,
        pattern: str | Pattern[str] | None = None,
        num_samples: int | None = None,
    ) -> ActivationData:
        """
        Collect activation statistics from layers matching the specified pattern.

        This method profiles the model by running inference on the dataset and collecting
        raw activation values at both the input and output of matching layers.

        Example of return format:
            {
                'layer_name_1': {
                    'in': [array1, array2, ...],   # List of input activation arrays
                    'out': [array1, array2, ...]   # List of output activation arrays
                },
                'layer_name_2': {...},
                ...
            }
            Each array is a numpy array containing activation values for one sample.

        :param model: OpenVINO model to profile
        :param dataset: Optional dataset to use for collection. If None, uses the instance's dataset.
            Should be compatible with NNCF's OVStatisticsAggregator
        :param pattern: Optional regex pattern to match layer names. If None, uses the instance's pattern.
            Examples: r'self_attn', r'__module.model.layers.\d+.mlp'
        :param num_samples: Optional number of samples to collect. If None, uses the instance's num_samples
        :return: Dictionary mapping layer names to their collected activations.
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

        # Find target nodes matching the pattern and list them in topological order
        target_ops = []
        for node in graph.topological_sort():
            if regexp.search(node.node_key):
                target_ops.append(node)

        if len(target_ops) == 0:
            msg = f"No layers found matching pattern: {pattern}"
            raise ValueError(msg)

        # Register statistic collection points and collect statistics
        statistic_points = self._get_statistic_points(model, graph, target_ops, num_samples)
        statistics_aggregator.register_statistic_points(statistic_points)
        statistics_aggregator.collect_statistics(model, graph)

        # Extract and convert collected statistics to numpy arrays
        result: ActivationData = defaultdict(dict)
        target_type_to_str = {
            TargetType.PRE_LAYER_OPERATION: "in",
            TargetType.POST_LAYER_OPERATION: "out",
        }
        for _, statistic_point, tensor_collector in statistic_points.get_tensor_collectors():
            if statistic_point.target_point.type not in target_type_to_str:
                msg = f"Unsupported target type: {statistic_point.target_point.type}"
                raise RuntimeError(msg)
            insert_type = target_type_to_str[statistic_point.target_point.type]
            layer_name = statistic_point.target_point.target_node_name
            stats = tensor_collector.get_statistics().values
            result[layer_name][insert_type] = [np.array(elem.data) for elem in stats]

        return result

    def calculate_stats(self, data: ActivationData, statistics: list[str] | None = None) -> pd.DataFrame:
        """
        Calculate custom statistics for collected activations using registered statistic functions.

        :param data: Activation data dictionary as returned by collect_activations().
            Format: {layer_name: {'in': [arrays...], 'out': [arrays...]}}
        :param statistics: List of statistic names to compute. If None, uses all registered statistics.
            Statistics must be registered in STATISTICS registry.
        :return: pandas DataFrame with columns:
            - name: Layer name
            - type: Activation type ('in' or 'out')
            - <statistic_name>: One column for each requested statistic

        Example:
            # Register custom statistic
            @NNCFProfiler.statistic("median")
            def median(vals):
                return float(np.median(vals))

            profiler.calculate_stats(acts, statistics=["min", "max", "median"])
        """
        # Use all registered statistics if not specified
        if statistics is None:
            statistics = list(self.STATISTICS.keys())

        # Validate all statistics are registered
        for stat in statistics:
            if stat not in self.STATISTICS:
                msg = f"Unknown statistic '{stat}'. Available: {list(self.STATISTICS.keys())}"
                raise ValueError(msg)

        activation_types = ["in", "out"]
        result_data = {"name": [], "type": []}

        # Initialize columns for each statistic
        for stat in statistics:
            result_data[stat] = []

        # Calculate statistics for each layer and activation type
        for layer_name in data:
            result_data["name"].extend([layer_name] * len(activation_types))
            result_data["type"].extend(activation_types)

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
        metrics: list[str] | None = None,
        statistics: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compare activations between two model variants using specified metrics and statistics.

        :param data1: Activation data from the first model (baseline)
        :param data2: Activation data from the second model (modified)
        :param metrics: List of comparator metric names to compute. If None, uses all registered comparators.
            Metrics must be registered in COMPARATORS registry.
        :param statistics: List of statistic names to compute for each dataset. If None, uses all registered statistics.
            Statistics must be registered in STATISTICS registry.
        :return: pandas DataFrame containing:
            - name: Layer name
            - type: Activation type ('in' or 'out')
            - <stat>_data1: Statistics for data1 (e.g., 'mean_data1', 'std_data1')
            - <stat>_data2: Statistics for data2 (e.g., 'mean_data2', 'std_data2')
            - <metric>: Comparator metrics (e.g., 'mean_diff', 'relative_diff')
        """
        # Verify both datasets have the same layers
        assert set(data1.keys()) == set(data2.keys()), "Activation datasets must contain the same layer names"

        # Use all registered comparators if metrics not specified
        if metrics is None:
            metrics = list(self.COMPARATORS.keys())

        # Use all registered statistics if not specified
        if statistics is None:
            statistics = list(self.STATISTICS.keys())

        # Validate all metrics are registered
        for metric in metrics:
            if metric not in self.COMPARATORS:
                msg = f"Unknown comparator '{metric}'. Available: {list(self.COMPARATORS.keys())}"
                raise ValueError(msg)

        # Validate all statistics are registered
        for stat in statistics:
            if stat not in self.STATISTICS:
                msg = f"Unknown statistic '{stat}'. Available: {list(self.STATISTICS.keys())}"
                raise ValueError(msg)

        # Calculate statistics for data1 and data2 using calculate_stats() method
        stats1_df = self.calculate_stats(data1, statistics=statistics)
        stats2_df = self.calculate_stats(data2, statistics=statistics)

        # Rename statistic columns to indicate which dataset they belong to
        for stat in statistics:
            stats1_df.rename(columns={stat: f"{stat}_data1"}, inplace=True)
            stats2_df.rename(columns={stat: f"{stat}_data2"}, inplace=True)

        # Merge statistics dataframes
        result = stats1_df.copy()
        for col in stats2_df.columns:
            if col not in ["name", "type"]:
                result[col] = stats2_df[col]

        # Calculate comparator metrics for each layer and activation type
        activation_types = ["in", "out"]

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
                assert len(vals1_list) == len(vals2_list), (
                    f"Number of samples mismatch for {layer_name} {activation_type}"
                )

                for i in range(len(vals1_list)):
                    assert vals1_list[i].shape == vals2_list[i].shape, (
                        f"Shape mismatch for {layer_name} {activation_type} sample {i}"
                    )

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

        :param plot_type: Name of the visualizer to use (must be registered in VISUALIZERS)
        :param args: Positional arguments to pass to the visualizer
        :param kwargs: Keyword arguments to pass to the visualizer
        :return: Return value from the visualizer function (typically a matplotlib figure)

        Example:
            profiler.plot("hist", acts_fp, acts_int, layer="layer_7")
            profiler.plot("mean_std", acts_fp)
            profiler.plot("metric", cmp, metric="relative_diff")
        """
        if plot_type not in self.VISUALIZERS:
            msg = f"Unknown plot type '{plot_type}'. Available: {list(self.VISUALIZERS.keys())}"
            raise ValueError(msg)
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
    activation_type: str | None = None,
    bins: int = 100,
    show_histograms: bool = True,
    show_summary: bool = True,
    display_figures: bool = False,
    data1_label: str = "data1",
    data2_label: str = "data2",
    **kwargs,
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

    :param data1: First activation dataset (e.g., FP32 model)
    :param data2: Second activation dataset (e.g., quantized model)
    :param activation_type: 'in', 'out', or None. If None, processes both 'in' and 'out' activations
    :param bins: Number of bins for histograms
    :param show_histograms: Whether to show individual histogram plots for each layer
    :param show_summary: Whether to show the summary plot with statistics
    :param display_figures: Whether to display figures immediately. If False, figures are only returned.
        Set to False to avoid memory issues with many figures.
    :param data1_label: Label for data1 in plots (e.g., 'fp16', 'fp32')
    :param data2_label: Label for data2 in plots (e.g., 'int8', 'fp8')
    :param kwargs: Additional arguments for plotting
    :return: List of matplotlib figures (histograms + summary plot)

    Example:
        # Compare both input and output activations
        figs = profiler.plot("compare_detailed", acts_fp32, acts_int8,
                             data1_label='FP32', data2_label='INT8',
                             display_figures=False)
        # Display or save figures as needed
        for i, fig in enumerate(figs):
            fig.savefig(f'comparison_{i}.png')
            plt.close(fig)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        msg = "matplotlib is required for visualization. Install with: pip install matplotlib"
        raise ImportError(msg)

    # Verify both datasets have the same layers
    assert set(data1.keys()) == set(data2.keys()), "Activation datasets must contain the same layer names"

    # Determine which activation types to process
    if activation_type is None:
        activation_types = ["in", "out"]
    else:
        activation_types = [activation_type]

    all_figures = {}
    summary_figures = {}
    layer_names = list(data1.keys())

    # Prepare data for summary plots (one per activation type)
    summary_data = {}
    for act_type in activation_types:
        summary_data[act_type] = {
            data1_label: {"mean": [], "std": []},
            data2_label: {"mean": [], "std": []},
            "diff": {"mean": [], "std": [], "relative": []},
        }

    # Loop over layers first, then activation types
    for layer_name in layer_names:
        all_figures[layer_name] = {}
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
                fig.suptitle(f"{layer_name} - {act_type}")

                ax.hist(data1_flat, bins=bins, alpha=0.5, label=f"{data1_label}")
                ax.hist(data2_flat, bins=bins, alpha=0.5, label=f"{data2_label}")
                ax.set_xlabel("Activation Value")
                ax.set_ylabel("Frequency")
                ax.legend(loc="best")
                ax.grid(True, alpha=0.3)

                # Add statistics text
                stats_text = (
                    f"{data1_label}: min={data1_flat.min():.4f}, max={data1_flat.max():.4f}\n"
                    f"{data2_label}: min={data2_flat.min():.4f}, max={data2_flat.max():.4f}"
                )
                ax.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

                all_figures[layer_name][act_type] = fig

                # Close figure if not displaying to save memory
                if not display_figures:
                    plt.close(fig)

            # Collect statistics for summary plot
            plot_data = summary_data[act_type]
            plot_data[data1_label]["mean"].append(np.mean(data1_flat))
            plot_data[data2_label]["mean"].append(np.mean(data2_flat))
            plot_data["diff"]["mean"].append(np.mean(data1_flat - data2_flat))
            plot_data["diff"]["relative"].append(np.mean(np.abs(data1_flat - data2_flat) / (np.abs(data1_flat) + 1e-8)))

            plot_data[data1_label]["std"].append(np.std(data1_flat))
            plot_data[data2_label]["std"].append(np.std(data2_flat))
            plot_data["diff"]["std"].append(np.std(data1_flat - data2_flat))

    # Create summary plots (one for each activation type)
    if show_summary:
        for act_type in activation_types:
            plot_data = summary_data[act_type]

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16))
            fig.suptitle(f"Activation Comparison Summary - {act_type}", fontsize=14)

            x = range(len(layer_names))

            # Plot 1: data1 mean and std
            ax1.errorbar(
                x,
                plot_data[data1_label]["mean"],
                plot_data[data1_label]["std"],
                linestyle="none",
                marker="^",
                capsize=5,
                **kwargs,
            )
            ax1.set_ylabel("Mean ± Std")
            ax1.set_title(f"{data1_label} Activations")
            ax1.set_xticks(x)
            ax1.set_xticklabels([str(i) for i in x], fontsize=9)
            ax1.grid(True, alpha=0.3)

            # Plot 2: data2 mean and std
            ax2.errorbar(
                x,
                plot_data[data2_label]["mean"],
                plot_data[data2_label]["std"],
                linestyle="none",
                marker="^",
                ecolor="green",
                capsize=5,
                **kwargs,
            )
            ax2.set_ylabel("Mean ± Std")
            ax2.set_title(f"{data2_label} Activations")
            ax2.set_xticks(x)
            ax2.set_xticklabels([str(i) for i in x], fontsize=9)
            ax2.grid(True, alpha=0.3)

            # Plot 3: difference mean and std
            ax3.errorbar(
                x,
                plot_data["diff"]["mean"],
                plot_data["diff"]["std"],
                linestyle="none",
                marker="^",
                ecolor="red",
                capsize=5,
                **kwargs,
            )
            ax3.set_ylabel("Mean ± Std")
            ax3.set_title(f"{data1_label} - {data2_label} Difference")
            ax3.set_xticks(x)
            ax3.set_xticklabels([str(i) for i in x], fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color="k", linestyle="--", alpha=0.3)

            # Plot 4: relative difference
            ax4.plot(x, plot_data["diff"]["relative"], color="green", marker="o", **kwargs)
            ax4.set_ylabel("Relative Difference")
            ax4.set_title(f"{data1_label} - {data2_label} Relative Difference")
            ax4.set_xlabel("Layer Index")
            ax4.set_xticks(x)
            ax4.set_xticklabels([str(i) for i in x], fontsize=9)
            ax4.grid(True, alpha=0.3)

            # Create legend with layer name mappings at the bottom
            legend_text = "Layer Mapping:\n" + ", ".join([f"{i}: {name}" for i, name in enumerate(layer_names)])
            fig.text(
                0.5,
                0.02,
                legend_text,
                ha="center",
                va="top",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                wrap=True,
            )

            # Adjust layout to make room for legend
            try:
                plt.tight_layout(rect=[0, 0.08, 1, 0.97])
            except Exception:
                plt.tight_layout()

            summary_figures[act_type] = fig

            # Close figure if not displaying to save memory
            if not display_figures:
                plt.close(fig)
        return all_figures, summary_figures
    return all_figures
