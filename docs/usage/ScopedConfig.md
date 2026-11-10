# Scoped Config

NNCF operates on an intermediate representation of a model graph called `NNCFGraph`.
To control how weight compression is applied to specific parts of a model, you can specify scoped configs - parts of the graph that should be compressed with a user-defined configuration.

Scoped configs select nodes by the same rules as [the ignored scope](/docs/usage/IgnoredScope.md) - node names, operation types, regular expressions, or entire subgraphs - and are passed to `nncf.compress_weights` via the `scoped_configs` parameter as a list of `nncf.ScopedConfig`.

## Usage

```python
import nncf

# Select by node name:
scoped_config = nncf.ScopedConfig(names=["node_name"], mode=nncf.CompressWeightsMode.INT8_ASYM)

# Select using regular expressions:
scoped_config = nncf.ScopedConfig(patterns=["node_\\d"], mode=nncf.CompressWeightsMode.INT8_ASYM)

# Select by operation type:
scoped_config = nncf.ScopedConfig(types=["MatMul"], mode=nncf.CompressWeightsMode.INT8_ASYM)

# Select by subgraph:
scoped_config = nncf.ScopedConfig(
    subgraphs=[nncf.Subgraph(inputs=["start_node"], outputs=["end_node"])],
    mode=nncf.CompressWeightsMode.INT8_ASYM,
)

# Set a custom group size:
scoped_config = nncf.ScopedConfig(patterns=["node_\\d"], mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=128)

compressed_model = nncf.compress_weights(
    model,
    mode=nncf.CompressWeightsMode.INT4_SYM,
    scoped_configs=[scoped_config],
)
```

- A matched node gets the configuration of the scoped config. If several scoped configs match the same node, the **last** one in the list wins.
- Scoped configs take precedence over the `ignored_scope`, `all_layers` and `backup_mode` options and over the precision assigned by the mixed precision algorithm.

See the [weight compression documentation](/docs/usage/post_training_compression/weights_compression/Usage.md#custom-precision-for-specific-layers) for more details.

## Inspecting and Visualizing the NNCFGraph

### Print all names and types of nodes in the NNCFGraph

```python
graph = nncf.build_graph(model)
for x in graph.get_all_nodes():
    print(f"Node name: {x.node_name}, Node type: {x.node_type}")
```

### Export the NNCFGraph to a DOT file

```python
graph = nncf.build_graph(model)
graph.dump_graph("model.dot")
```

### Convert dot to svg file

```sh
python tools/other/render_dot_to_svg.py -i model.dot
```
