import pytest

def pytest_addoption(parser):
    parser.addoption("--data", action="store", default="/mnt/datasets/imagenet/val")