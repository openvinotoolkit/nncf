import pytest
import numpy as np
from pathlib import Path

def pytest_addoption(parser):
    parser.addoption("--data", action="store")
    parser.addoption("--output", action="store", default="./tmp/")

def pytest_configure(config):
    config.test_results = []

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()

    if result.when == 'call':
        table = item.config.test_results
        header = ["Model name", "FP32 top-1", "Torch INT8 top-1", "ONNX INT8 top-1", "OV INT8 top-1",
            "FP32 FPS", "Torch INT8 FPS", "ONNX INT8 FPS", "OV INT8 FPS"]
        output_folder = Path(item.config.getoption("--output"))
        output_folder.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_folder / "results.csv", table, delimiter=",", fmt='%s', header=','.join(header))
