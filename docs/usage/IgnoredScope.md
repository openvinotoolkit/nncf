# Ignored Scope

NNCF operates on an intermediate representation of a model graph called `NNCFGraph`.
To control how model compression algorithms are applied, you can specify an ignored scope - parts of the graph that should be excluded from compression.

Ignored scopes are defined using node names, operation types, regular expressions, or entire subgraphs and are passed to compression algorithms via nncf.IgnoredScope.

## Usage

```python
import nncf

# Exclude by node name:
ignored_scope = nncf.IgnoredScope(names=["node_name"])

# Exclude using regular expressions:
ignored_scope = nncf.IgnoredScope(patterns=["node_\\d"])

# Exclude by operation type:
ignored_scope = nncf.IgnoredScope(types=["Multiply"])

# Exclude by subgraph
ignored_scope = nncf.IgnoredScope(
    subgraphs=nncf.Subgraph(inputs=["start_node"], outputs=["end_node"])
)
```

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
python tools/render_dot_to_svg.py -m model.dot
```
