# AGENTS.md - PyTorch backend

Backend for eager `torch.nn.Module`. General rules: [root AGENTS.md](../../../AGENTS.md).
For `torch.fx.GraphModule` see [TorchFX AGENTS.md](../experimental/torch/fx/AGENTS.md).

## Key Modules

- `function_hook/` - current model-modification mechanism:
  - `wrapper.py` - `wrap_model`, `ForwardWithHooks`, `register_pre/post_function_hook`
  - `hook_storage.py` - `HookStorage` holding inserted compression modules
  - `nncf_graph/nncf_graph_builder.py` - `build_nncf_graph`, `GraphModelWrapper`
  - `model_transformer.py` - `PT2ModelTransformer`
  - `quantization/quantize_model.py` - `quantize_impl`, `compress_weights_impl`
  - `serialization.py` - `get_config` / `load_from_config`
- `graph/operator_metatypes.py` - `PT_OPERATOR_METATYPES` (also reused by TorchFX)
- `graph/transformations/commands.py` - `PTTargetPoint` and commands
- `strip.py` - `nncf.torch.strip`
- `extensions/` - C++/CUDA kernels

## Hooks: Insertion and Execution

- `wrap_model` replaces `model.forward` with `ForwardWithHooks` and adds an empty `HookStorage` to the model.
- Hooks are `nn.Module`s registered via `register_pre/post_function_hook(model, op_name, port_id, hook)`.
- Algorithms insert hooks via `PT2InsertionCommand` + `PT2ModelTransformer`; `op_name` is the `NNCFGraph` node name.
- On forward, `FunctionHookMode` (`TorchFunctionMode`) intercepts each torch function call and builds its `op_name`.
- It runs pre-hooks on inputs, calls the function, then runs post-hooks on outputs.

## Gotchas

- Compression ops are inserted as hooks into `HookStorage` of a wrapped model, not by replacing modules.
- `nncf.quantize` deep-copies the input model; `PT2ModelTransformer` then modifies the wrapped model in place.
- Graph is traced from a real forward pass, so an example input is required to build `NNCFGraph`.
- Serialization: compressed state is restored via `nncf.torch.load_from_config` + `load_state_dict`.

## Tests

- `tests/torch/` (markers `cuda`, `long`): `make test-torch-cpu`, `make test-torch-gpu`, `make test-torch-long`.
- Hook mechanism tests: `tests/torch/function_hook/`; reference data: `tests/torch/data/function_hook/`.
- Regenerate references: `pytest tests/torch/... --regen-ref-data`.
