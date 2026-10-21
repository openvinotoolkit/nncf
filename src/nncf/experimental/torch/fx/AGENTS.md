# AGENTS.md - TorchFX backend (experimental)

Backend for `torch.fx.GraphModule` produced by `torch.export`. General rules: [root AGENTS.md](../../../../../AGENTS.md).

## Key Modules

- `nncf_graph_builder.py` - `GraphConverter`: `GraphModule` -> `NNCFGraph`
- `model_transformer.py` - `FXModelTransformer`; `commands.py` - `FXApplyTransformationCommand`, `FXModelExtractionCommand`
- `transformations.py` - graph-level FX transformations used by commands
- `quantization/quantize_model.py` - `quantize_impl`, `compress_weights_impl` (used by `nncf.quantize`)
- `quantization/quantize_pt2e.py` - `quantize_pt2e` for torchao/PT2E quantizers; `quantization/quantizer/` - quantizer adapters
- `constant_folding.py` - constant folding pass

## Gotchas

- Operator metatypes are shared with PyTorch: `src/nncf/torch/graph/operator_metatypes.py`.
- Input must be an exported `GraphModule` (`torch.export.export(...).module()`), not an eager `nn.Module`.
- `quantize_impl` deep-copies the model and rebuilds `GraphModule` after transformations; keep `model.meta` intact.
- Not all algorithms are supported; check the `*_impl` functions for rejected parameters before adding features.
- The whole `src/nncf/experimental` folder is excluded from mypy.

## Tests

- `tests/torch/fx/` (part of `make test-torch-cpu`); reference data: `tests/torch/data/fx/`.
- ExecuTorch/quantizer integration: `tests/executorch/`.
- Regenerate references: `pytest tests/torch/fx/... --regen-ref-data`.
