"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import glob
import stat
import sys
import sysconfig

import codecs
import os
import re
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open("{}/README.md".format(here), "r") as fh:
    long_description = fh.read()


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

INSTALL_REQUIRES = ["ninja>=1.10.0.post2",
                    "addict>=2.4.0",
                    "texttable>=1.6.3",
                    "scipy>=1.3.2",
                    "networkx>=2.5",
                    "graphviz>=0.15",
                    "jsonschema>=3.2.0",
                    "pydot>=1.4.1",
                    "jstyleson>=0.0.2",
                    "numpy>=1.19",
                    "tqdm>=4.54.1",
                    "natsort>=7.1.0",
                    "pandas>=1.1.5",
                    "scikit-learn>=0.24.0",
                    "wheel>=0.36.1"]

python_version = sys.version_info[:2]
if python_version < (3, 6):
    print("Only Python >= 3.6 is supported")
    sys.exit(0)

version_string = "{}{}".format(sys.version_info[0], sys.version_info[1])

if "--cpu-only" in sys.argv:
    print("WARNING: --cpu-only option for NNCF setup.py is deprecated and will "
          "be ignored. CPU/GPU support will be determined by the torch package.")
    sys.argv.remove("--cpu-only")

if not os.environ.get('NNCF_SKIP_INSTALLING_TORCH'):
    INSTALL_REQUIRES.extend(["torch>=1.5.0, <=1.8.1, !=1.8.0"])
else:
    print("Skipping torch installation for NNCF.")
    DEPENDENCY_LINKS = []


EXTRAS_REQUIRE = {
    "tests": [
        "pytest"],
    "docs": []
}

setup(
    name="nncf",
    version=find_version(os.path.join(here, "nncf/version.py")),
    author="Intel",
    author_email="alexander.kozlov@intel.com",
    description="Neural Networks Compression Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/openvinotoolkit/nncf_pytorch",
    packages=find_packages(exclude=["tests", "tests.*",
                                    "examples", "examples.*",
                                    "tools", "tools.*",
                                    "beta", "beta.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    keywords=["compression", "quantization", "sparsity", "mixed-precision-training",
              "quantization-aware-training", "hawq", "classification",
              "pruning", "object-detection", "semantic-segmentation", "nlp",
              "bert", "transformers", "mmdetection"],
    include_package_data=True
)

path_to_ninja = glob.glob(str(sysconfig.get_paths()["purelib"]+"/ninja*/ninja/data/bin/"))
if path_to_ninja:
    path_to_ninja = str(path_to_ninja[0]+"ninja")
    if not os.access(path_to_ninja, os.X_OK):
        st = os.stat(path_to_ninja)
        os.chmod(path_to_ninja, st.st_mode | stat.S_IEXEC)
