import os
import site
from pathlib import Path

TEST_ROOT = Path(__file__).absolute().parents[1]
PROJECT_ROOT = TEST_ROOT.parent.absolute()  # pylint:disable=no-member
EXAMPLES_DIR = PROJECT_ROOT / "examples"
GITHUB_REPO_URL = "https://github.com/openvinotoolkit/nncf/"


def _get_site_packages_path() -> Path:
    site_packages = site.getsitepackages()
    return Path(next(x for x in site_packages if "lib" in x and "site-packages" in x))


DATASET_DEFINITIONS_PATH = _get_site_packages_path() / "openvino" / "model_zoo" / "data" / "dataset_definitions.yml"

ROOT_PYTHONPATH_ENV = os.environ.copy().update({"PYTHONPATH": str(PROJECT_ROOT)})
