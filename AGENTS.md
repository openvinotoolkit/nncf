# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

Neural Network Compression Framework (NNCF) provides post-training and training-time compression algorithms
(quantization, weight compression, pruning, QAT) for OpenVINO, PyTorch, TorchFX, and ONNX models.
Public API entry points: `nncf.quantize`, `nncf.compress_weights`, `nncf.Dataset`.

## Architecture

Backend-agnostic core built around `NNCFGraph`. Details: [docs/NNCFArchitecture.md](docs/NNCFArchitecture.md).

```text
src/nncf/
├── common/                        # Backend-agnostic logic: graph, quantization, statistics
├── quantization/algorithms/<algo> # Algorithm logic + <backend>_backend.py implementations
├── quantization/quantize_model.py # Public API, dispatches to backend *_impl functions
├── tensor/                        # Backend-agnostic tensor and functions
├── openvino/  onnx/  torch/       # Backend-specific code (see AGENTS.md in each folder)
└── experimental/torch/fx/         # TorchFX backend (experimental)
```

Folder-specific guidance:
[Common](src/nncf/common/AGENTS.md),
[OpenVINO](src/nncf/openvino/AGENTS.md),
[PyTorch](src/nncf/torch/AGENTS.md),
[ONNX](src/nncf/onnx/AGENTS.md),
[TorchFX](src/nncf/experimental/torch/fx/AGENTS.md).

Rules:

- Common algorithm logic goes in `algorithm.py`; backend interface in `backend.py`; backend specifics in
  `openvino_backend.py`, `onnx_backend.py`, `torch_backend.py`, `torch_fx_backend.py`
  (see [min_max](src/nncf/quantization/algorithms/min_max) as a reference).
- Do not import backend frameworks (`openvino`, `torch`, `onnx`) at module level in `src/nncf/common` or in
  algorithm common code; backend dispatch uses local imports (see [factory.py](src/nncf/common/factory.py)).
- Use `nncf.tensor` functions instead of raw numpy/torch calls in backend-agnostic code.
- A change in common code affects all backends: run tests for every supported backend.

## Build and Test

Each test target needs the matching install target first. `install-<backend>` installs the test requirements
and, for most targets, NNCF in editable mode (`uv pip install -e .`); see [Makefile](Makefile).

```bash
make install-<backend>   # common | openvino | torch | onnx | tools | docs | examples | conformance
make test-<backend>      # common | openvino | torch | onnx | tools | docs | examples
make install-pre-commit
```

Backend-specific test targets:

```bash
make test-openvino       # runs with ONEDNN_MAX_CPU_ISA=AVX2 OV_CPU_MAX_ISA=AVX2
make test-torch-cpu      # excludes `cuda` and `long` markers
make test-torch-gpu      # `cuda` marker only
make test-ptq            # conformance, requires install-conformance
NUM_WORKERS=4 make test-torch-cpu ARGS="-k test_name"

python -m pytest tests/<backend>/path/test_file.py::test_name
```

- Regenerate reference data (graphs, JSON refs): `pytest tests/<backend>/... --regen-ref-data`. Review the diff.
- Shared test logic lives in [tests/cross_fw/test_templates](tests/cross_fw/test_templates);
  backend tests subclass these templates.
- Lint and format: `make pre-commit` (ruff, markdownlint, cspell, actionlint).
- Type check: `mypy` (strict, `src/nncf`; excluded files are listed in [pyproject.toml](pyproject.toml)).

## Code Style

Full guide: [docs/styleguide/PyGuide.md](docs/styleguide/PyGuide.md).

- Type hints (PEP-484) for all functions, including tests and fixtures.
- No `assert` in `src/nncf`; raise exceptions with informative messages (see `nncf.errors`).
- Use `pathlib.Path` instead of `os.path`.
- Naming: `lower_with_under` for modules/functions/variables, `CapWords` for classes, `CAPS_WITH_UNDER` for
  constants, `_` prefix for private. Avoid type suffixes in names (`id_to_name_dict`) and generic module names
  (`utils.py`, `helpers.py`). Avoid single-char names except `i j k v` (loops), `e` (exceptions), `f` (files),
  `a b x y z` (tensors).
- Use nested functions/classes only when they simplify code or close over local variables.
- Externally visible functions need docstrings unless very short and obvious.
- Every `.py` file starts with the Intel Apache-2.0 copyright header (ruff `CPY001`); copy it from any existing file.
- Docstring format:

```python
def foo(arg1: int, arg2: bool = False) -> int:
    """
    Brief description of what the function does.

    :param arg1: Description
    :param arg2: Description
    :return: Description
    """
```

## Tests

- pytest only; use fixtures, not xunit-style setup/teardown.
- Place tests in `tests/<backend>/` with `test_*` file names; test data in `tests/<backend>/data/`.
- Markers are defined in `tests/<backend>/pytest.ini` (e.g. `cuda`, `long` for torch).

## Related Docs

- [docs/usage](docs/usage) - algorithm documentation; [examples](examples) - per-backend examples
- [CONTRIBUTING.md](CONTRIBUTING.md) - contribution workflow
- [AI_POLICY.md](AI_POLICY.md) - rules for AI-assisted contributions
- [.github/copilot-instructions.md](.github/copilot-instructions.md) - code review guidelines
- [.github/skills](.github/skills) - task-specific agent skills (release notes, dependency checks, workflows)
