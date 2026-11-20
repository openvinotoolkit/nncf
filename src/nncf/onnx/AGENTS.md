# AGENTS.md - ONNX backend

Backend for `onnx.ModelProto`. General rules: [root AGENTS.md](../../../AGENTS.md).

## Key Modules

- `graph/nncf_graph_builder.py` - `GraphConverter`: `onnx.ModelProto` -> `NNCFGraph`, `ONNXLayerAttributes`
- `graph/metatypes/onnx_metatypes.py` - `ONNX_OPERATION_METATYPES`, `ONNXOpMetatype`, `ONNXOpWithWeightsMetatype`
- `graph/model_transformer.py` - `ONNXModelTransformer` (inserts QuantizeLinear/DequantizeLinear pairs)
- `graph/transformations/commands.py` - `ONNXTargetPoint`, `ONNXInsertionCommand` and other commands
- `quantization/quantize_model.py` - `quantize_impl`, `quantize_with_accuracy_control_impl`, `compress_weights_impl`
- `engine.py` - `ONNXEngine` (onnxruntime) for inference during calibration

## Gotchas

- Models are protobuf objects; `ONNXModelTransformer` works on a `deepcopy` unless `inplace=True`.
- Weight/bias port detection depends on metatype definitions; check `graph/metatypes/groups.py` when adding ops.
- Models over 2GB need external data; see `check_model_protobuf_size` / `check_external_data_location`.

## Tests

- `tests/onnx/`, run with `make test-onnx`.
- Reference graphs: `tests/onnx/data/reference_graphs/`; regenerate with `pytest tests/onnx/... --regen-ref-data`.
