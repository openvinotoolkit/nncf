# Calibration tool for testing OpenVINO backend using POT config

## How to run

The `calibrate.py` supports `pot` and `native` implementation of the OpenVINO backend. The implementation should be specified using `--impl` command line argument.

```bash
python calibrate.py \
  --config <path to POT config> \
  --output-dir <path to output dir> \
  --impl pot
```
