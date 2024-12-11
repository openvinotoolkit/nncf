# Torch compile with OpenVino backend performance check

The [main.py](main.py) script checks fp32 and int8 models performance in two setups:

* Compilation via `torch.compile(model, backend="openvino")`
* Export to OpenVino via `torch.export.export` + `ov.convert` functions

## Installation

```bash
# From the root of NNCF repo:
make install-torch-test
pip install -r tests/torch/fx/performance_check/requirements.txt
```

## Usage

Run performance check for all models:

```bash
python main.py
```

Run performance check for a specific model:

```bash
python main.py --model model_name
```

Run performance check for a specific model and save performance check result to a specific location:

```bash
python main.py --model model_name --file_name /path/to/save/resuts.csv
```

Names of the available models could be found in [model_scope.py](model_scope.py) as keys of the `MODEL_SCOPE` dict.
Performance check results are saved to a `result.csv` file by default.

## Artefacts

You will find directories named after the models in current directory. In case errors were not occured during the preformance check, each directory should contain:

* `int8_code.py` - code of the quantized torch.fx.GrpahModule model

* `int8_nncf_graph.dot` - nncf graph visualization of the quantized torch.fx.GrpahModule model

* `result.csv` - results of the performance check the current model.
