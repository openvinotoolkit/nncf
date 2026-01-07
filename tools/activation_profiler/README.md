## Profiler

The Profiler is a tool for collecting and analyzing activation statistics from OpenVINO models. It enables
layer-by-layer profiling of model activations using NNCF infrastructure, making it useful for debugging quantization
and compression issues, comparing model variants, and understanding activation distributions.

Key features:

- Collect raw activations at input and output of specific layers using regex pattern matching
- Calculate custom statistics (min, max, mean, std, percentiles, etc.) on collected activations
- Compare activations between two model variants (e.g., FP32 vs INT8) with built-in and custom metrics
- Visualize activation distributions and comparison results with matplotlib
- Extensible architecture allowing registration of custom statistics, comparators, and visualizers

See [nncf_profiler_example.ipynb](nncf_profiler_example.ipynb) for a complete usage example demonstrating how
to profile an OpenVINO model, collect activation statistics, and compare FP32 vs INT8 quantized variants.
