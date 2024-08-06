# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# *WARNING*: Do not run this file directly by `python setup.py install`
# or with any other parameter - this is an outdated and error-prone way
# to install Python packages and causes particular problems with namespace
# packages such as `protobuf`.
# Refer to the table below for well-known and supported alternatives with
# the same behavior:
# +-------------------------------------+---------------------------------------------+
# |           Old command               |                New command                  |
# +-------------------------------------+---------------------------------------------+
# | python setup.py install             | pip install .                               |
# | python setup.py develop             | pip install -e .                            |
# | python setup.py develop --*arg*     | pip install --install-option="*arg*" -e  .  |
# | python setup.py sdist               | python -m build -s                          | <-- using the "build" package
# | python setup.py bdist_wheel         | python -m build -w                          | <-- pypi.org/project/build/
# | python setup.py bdist_wheel --*arg* | python -m build -w -C--global-option=--*arg*|
# +-------------------------------------+---------------------------------------------+
#
# PyPA in general recommends to move away from setup.py and use pyproject.toml
# instead. This doesn't fit us as we currently want to do custom stuff during
# installation such as setting version based on the commit SHA for repo-based
# installs.


import codecs
import glob
import os
import re
import stat
import sys
import sysconfig

import setuptools
from pkg_resources import parse_version
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
BKC_SETUPTOOLS_VERSION = "59.5.0"

setuptools_version = parse_version(setuptools.__version__).base_version
if setuptools_version < "43.0.0":
    raise RuntimeError(
        "To properly install NNCF, please install setuptools>=43.0.0, "
        f"while current setuptools version is {setuptools.__version__}. "
        f"Recommended version is {BKC_SETUPTOOLS_VERSION}."
    )

python_version = sys.version_info
if python_version < (3, 8, 0):
    print("Only Python >= 3.8.0 is supported")
    sys.exit(0)

version_string = "{}{}".format(sys.version_info[0], sys.version_info[1])

is_installing_editable = "develop" in sys.argv
is_building_release = not is_installing_editable and "--release" in sys.argv
if "--release" in sys.argv:
    sys.argv.remove("--release")


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if not version_match:
        raise RuntimeError("Unable to find version string.")
    version_value = version_match.group(1)
    if not is_building_release:
        if is_installing_editable:
            return version_value + ".dev0+editable"
        import subprocess  # nosec

        dev_version_id = "unknown_version"
        try:
            repo_root = os.path.dirname(os.path.realpath(__file__))
            dev_version_id = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)  # nosec
                .strip()
                .decode()
            )
        except subprocess.CalledProcessError:
            pass
        return version_value + f".dev0+{dev_version_id}"

    return version_value


INSTALL_REQUIRES = [
    "jsonschema>=3.2.0",
    "jstyleson>=0.0.2",
    "natsort>=7.1.0",
    "networkx>=2.6, <=3.3",
    "ninja>=1.10.0.post2, <1.12",
    "numpy>=1.19.1, <1.27",
    "openvino-telemetry>=2023.2.0",
    "packaging>=20.0",
    "pandas>=1.1.5,<2.3",
    "psutil",
    "pydot>=1.4.1, <3.0.0",
    "pymoo>=0.6.0.1",
    "rich>=13.5.2",
    "scikit-learn>=0.24.0",
    "scipy>=1.3.2",
    "tabulate>=0.9.0",
    "tqdm>=4.54.1",
]

EXTRAS_REQUIRE = {
    "plots": [
        "kaleido>=0.2.1",
        "matplotlib>=3.3.4, <3.6",
        "pillow>=9.0.0",
        "plotly-express>=0.4.1",
    ],
}

with open("{}/README.md".format(here), "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    name="nncf",
    version=find_version(os.path.join(here, "nncf/version.py")),
    author="Intel",
    author_email="alexander.kozlov@intel.com",
    description="Neural Networks Compression Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/openvinotoolkit/nncf",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "tools", "tools.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    keywords=[
        "compression",
        "quantization",
        "sparsity",
        "mixed-precision-training",
        "quantization-aware-training",
        "hawq",
        "classification",
        "pruning",
        "object-detection",
        "semantic-segmentation",
        "nas",
        "nlp",
        "bert",
        "transformers",
        "mmdetection",
    ],
    include_package_data=True,
)

path_to_ninja = glob.glob(str(sysconfig.get_paths()["purelib"] + "/ninja*/ninja/data/bin/"))
if path_to_ninja:
    path_to_ninja = str(path_to_ninja[0] + "ninja")
    if not os.access(path_to_ninja, os.X_OK):
        st = os.stat(path_to_ninja)
        os.chmod(path_to_ninja, st.st_mode | stat.S_IEXEC)
