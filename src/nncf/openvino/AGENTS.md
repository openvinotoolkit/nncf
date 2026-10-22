# AGENTS.md - OpenVINO backend

Backend for `openvino.Model`. General rules: [root AGENTS.md](../../../AGENTS.md).

## Key Modules

- `graph/nncf_graph_builder.py` - `GraphConverter`: `ov.Model` -> `NNCFGraph`
- `graph/metatypes/openvino_metatypes.py` - `OV_OPERATOR_METATYPES` registry, `OV*Metatype` classes
- `graph/model_transformer.py` - `OVModelTransformer`
- `graph/transformations/commands.py` - `OVTargetPoint`, `OVInsertionCommand` and other commands
- `quantization/quantize_model.py` - `quantize_impl`, `quantize_with_accuracy_control_impl`, `compress_weights_impl`
- `engine.py` - `OVNativeEngine` / `OVCompiledModelEngine` for inference during calibration
- `statistics/` - statistic collectors

## Gotchas

- `OVModelTransformer.transform` clones the model unless `inplace=True`; do not assume the input model is modified.
- Target points address explicit input/output ports (`port_id`), not just nodes.
- Preserve NNCF runtime info when creating new models (`copy_rt_info` in `graph/model_utils.py`).

## Tests

- `tests/openvino/native/`, run with `make test-openvino` (forces `OV_CPU_MAX_ISA=AVX2`, `ONEDNN_MAX_CPU_ISA=AVX2`).
- Reference data is versioned by OpenVINO release in `tests/openvino/native/data/<ov_version>/`;
  `get_actual_reference_for_current_openvino()` picks the newest version not above the installed one.
- Regenerate references: `pytest tests/openvino/native/... --regen-ref-data`.
