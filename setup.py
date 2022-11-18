"""
 Copyright (c) 2022 Intel Corporation
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
proper_version = '59.5.0'

with open("{}/README.md".format(here), "r", encoding="utf8") as fh:
    long_description = fh.read()

if "--tf" in sys.argv:
    import setuptools
    from pkg_resources import parse_version
    setuptools_version = parse_version(setuptools.__version__).base_version
    if setuptools_version < '43.0.0':
        raise RuntimeError(
            "To properly install NNCF, please install setuptools>=43.0.0, "
            f"while current setuptools version is {setuptools.__version__}. "
            f"Recommended version is {proper_version}."
        )


is_installing_editable = "develop" in sys.argv
is_building_release = not is_installing_editable and "--release" in sys.argv
if "--release" in sys.argv:
    sys.argv.remove("--release")


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
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
            dev_version_id = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],   # nosec
                                                     cwd=repo_root).strip().decode()
        except subprocess.CalledProcessError:
            pass
        return version_value + f".dev0+{dev_version_id}"

    return version_value


INSTALL_REQUIRES = ["ninja>=1.10.0.post2, <1.11",
                    "addict>=2.4.0",
                    "texttable>=1.6.3",
                    "scipy>=1.3.2, <=1.9.1",
                    "matplotlib>=3.3.4, <3.6",
                    "networkx>=2.6, <=2.8.2",  # see ticket 94048 or https://github.com/networkx/networkx/issues/5962
                    "numpy>=1.19.1, <1.24",
                    "pillow>=9.0.0",

                    # The recent pyparsing major version update seems to break
                    # integration with networkx - the graphs parsed from current .dot
                    # reference files no longer match against the graphs produced in tests.
                    # Using 2.x versions of pyparsing seems to fix the issue.
                    # Ticket: 69520
                    "pyparsing<3.0",
                    "pymoo==0.5.0",
                    "jsonschema==3.2.0",
                    "pydot>=1.4.1",
                    "jstyleson>=0.0.2",
                    "tqdm>=4.54.1",
                    "natsort>=7.1.0",
                    "pandas>=1.1.5,<1.4.0rc0",
                    "scikit-learn>=0.24.0",
                    "wheel>=0.36.1"]


python_version = sys.version_info
if python_version < (3, 7, 0):
    print("Only Python >= 3.7.0 is supported")
    sys.exit(0)

version_string = "{}{}".format(sys.version_info[0], sys.version_info[1])

EXTRAS_REQUIRE = {
    "tests": ["pytest"],
    "docs": [],
    "tf": [
        "tensorflow~=2.8.2",
    ],
    "torch": [
        "torch==1.12.1",
        # Please see
        # https://stackoverflow.com/questions/70520120/attributeerror-module-setuptools-distutils-has-no-attribute-version
        "setuptools==59.5.0"
    ],
    "onnx": [
        "torch==1.12.1",
        "torchvision==0.13.1",
        "onnx==1.12.0",
        "protobuf==3.20.1",
        "onnxruntime-openvino==1.13.1",
    ],
    "openvino": [
        "openvino-dev"
    ]
}

EXTRAS_REQUIRE["all"] = [
    EXTRAS_REQUIRE["tf"],
    EXTRAS_REQUIRE["torch"]
]

SETUP_REQUIRES = []

if "--torch" in sys.argv:
    INSTALL_REQUIRES.extend(EXTRAS_REQUIRE["torch"])
    sys.argv.remove("--torch")

if "--tf" in sys.argv:
    # See also: https://github.com/tensorflow/tensorflow/issues/56077
    # This is a temporary patch, that limites the protobuf version from above
    INSTALL_REQUIRES.extend(["protobuf>=3.9.2,<3.20"])
    INSTALL_REQUIRES.extend(EXTRAS_REQUIRE["tf"])
    sys.argv.remove("--tf")

if "--onnx" in sys.argv:
    INSTALL_REQUIRES.extend(EXTRAS_REQUIRE["onnx"])
    sys.argv.remove("--onnx")

if "--openvino" in sys.argv:
    INSTALL_REQUIRES.extend(EXTRAS_REQUIRE["openvino"])
    SETUP_REQUIRES.extend(EXTRAS_REQUIRE["openvino"])
    sys.argv.remove("--openvino")

if "--all" in sys.argv:
    INSTALL_REQUIRES.extend(EXTRAS_REQUIRE["all"])
    sys.argv.remove("--all")

setup(
    name="nncf",
    version=find_version(os.path.join(here, "nncf/version.py")),
    author="Intel",
    author_email="alexander.kozlov@intel.com",
    description="Neural Networks Compression Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/openvinotoolkit/nncf",
    packages=find_packages(exclude=["tests", "tests.*",
                                    "examples", "examples.*",
                                    "tools", "tools.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=INSTALL_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    keywords=["compression", "quantization", "sparsity", "mixed-precision-training",
              "quantization-aware-training", "hawq", "classification",
              "pruning", "object-detection", "semantic-segmentation", "nas", "nlp",
              "bert", "transformers", "mmdetection"],
    include_package_data=True
)

path_to_ninja = glob.glob(str(sysconfig.get_paths()["purelib"]+"/ninja*/ninja/data/bin/"))
if path_to_ninja:
    path_to_ninja = str(path_to_ninja[0]+"ninja")
    if not os.access(path_to_ninja, os.X_OK):
        st = os.stat(path_to_ninja)
        os.chmod(path_to_ninja, st.st_mode | stat.S_IEXEC)
