import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--sota-data-dir", type=str, default=None, help="Path to datasets directory for sota accuracy test"
    )


@pytest.fixture(scope="module")
def sota_data_dir(request):
    return request.config.getoption("--sota-data-dir")
