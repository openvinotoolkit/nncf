# AGENTS.md - Common (backend-agnostic) code

Shared core used by all backends. General rules: [root AGENTS.md](../../../AGENTS.md).

## Key Modules

- `graph/graph.py` - `NNCFGraph`; `graph/layer_attributes.py` - `BaseLayerAttributes` and subclasses
- `graph/operator_metatypes.py` - `OperatorMetatype`, `OperatorMetatypeRegistry` (base for backend metatypes)
- `graph/model_transformer.py` - `ModelTransformer` base
- `graph/transformations/` - `TargetPoint`, `TargetType`, `TransformationCommand`, `TransformationLayout`
- `graph/patterns/` - fusing/ignored patterns, `PatternsManager`
- `factory.py` - backend dispatch: `build_graph`, `ModelTransformerFactory`, `EngineFactory`,
  `CommandCreatorFactory`, `StatisticsAggregatorFactory`
- `utils/backend.py` - `BackendType`, `get_backend`, `is_<backend>_model` predicates
- `tensor_statistics/` - `TensorCollector`, reducers/aggregators, `StatisticPoint`, `StatisticPointsContainer`,
  `StatisticsAggregator` base
- `quantization/` - quantizer setup, propagation, configs (`structs.py`, `quantizer_propagation/`)
- `hardware/` - `HWConfig`, per-device setups in `setups/` (CPU, GPU, NPU)
- `logging/logger.py` - `nncf_logger`; `deprecation.py` - `deprecated`, `warning_deprecated`
- `utils/api_marker.py` - `@api` decorator for public API

## Rules

- No module-level imports of `torch`, `openvino`, `onnx`; import them locally inside backend-dispatch branches.
- Backend logic goes into backend folders; add a new backend branch to every factory in `factory.py`.
- Use `nncf.tensor` (`Tensor`, `nncf.tensor.functions`) for tensor math, not numpy/torch directly.
- Log via `nncf_logger`, not `print`; deprecate public API with `deprecation.py` helpers.
- Changes here affect every backend: run `make test-common` and the tests of all backends.

## Tests

- `tests/common/` (no backend frameworks required): `make install-common`, `make test-common`.
- Reference data: `tests/common/data/`; regenerate with `pytest tests/common/... --regen-ref-data`.
