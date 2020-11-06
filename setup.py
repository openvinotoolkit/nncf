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


INSTALL_REQUIRES = ["ninja",
                    "addict",
                    "pillow",
                    "texttable",
                    "scipy==1.3.2",
                    "networkx",
                    "graphviz",
                    "jsonschema",
                    "pydot",
                    "tensorboardX",
                    "jstyleson",
                    "matplotlib",
                    "numpy",
                    "tqdm",
                    "onnx",
                    "opencv-python",
                    "pytest-mock",
                    "prettytable",
                    "mdutils",
                    "yattag",
                    "jsonschema",
                    "wheel",
                    "defusedxml"]

DEPENDENCY_LINKS = []

python_version = sys.version_info[:2]
if python_version[0] < 3:
    print("Only Python > 3.5 is supported")
    sys.exit(0)
elif  python_version[1] < 5:
    print("Only Python > 3.5 is supported")
    sys.exit(0)

version_string = "{}{}".format(sys.version_info[0], sys.version_info[1])

INSTALL_REQUIRES.extend(["torch", "torchvision"])

TORCH_VERSION = "1.5.0"
TORCHVISION_VERSION = "0.6.0"
CUDA_VERSION = "102"
IS_CUDA_VER_DEFAULT_FOR_CURRENT_TORCH_VER = True

TORCH_SOURCE_URL_TEMPLATE = 'https://download.pytorch.org/whl/{mode}/torch-{tv}{whl_mode}-cp{ver}-cp{' \
                            'ver}m-linux_x86_64.whl'
TORCHVISION_SOURCE_URL_TEMPLATE = 'https://download.pytorch.org/whl/{mode}/torchvision-{tvv}{whl_mode}-cp{ver}-cp{' \
                                  'ver}m-linux_x86_64.whl'
WHL_MODE_TEMPLATE = '%2B{mode}'

if "--cpu-only" in sys.argv:
    mode = 'cpu'
    whl_mode = WHL_MODE_TEMPLATE.format(mode=mode)
    DEPENDENCY_LINKS = [
        TORCH_SOURCE_URL_TEMPLATE.format(
            tv=TORCH_VERSION,
            ver=version_string,
            mode=mode,
            whl_mode=whl_mode),
        TORCHVISION_SOURCE_URL_TEMPLATE.format(
            tvv=TORCHVISION_VERSION,
            ver=version_string,
            mode=mode,
            whl_mode=whl_mode)]
    sys.argv.remove("--cpu-only")
else:
    mode = "cu{}".format(CUDA_VERSION)
    whl_mode = '' if IS_CUDA_VER_DEFAULT_FOR_CURRENT_TORCH_VER else WHL_MODE_TEMPLATE.format(mode=mode)
    DEPENDENCY_LINKS = [
        TORCH_SOURCE_URL_TEMPLATE.format(
            tv=TORCH_VERSION,
            ver=version_string,
            mode=mode,
            whl_mode=whl_mode),
        TORCHVISION_SOURCE_URL_TEMPLATE.format(
            tvv=TORCHVISION_VERSION,
            ver=version_string,
            mode=mode,
            whl_mode=whl_mode)]


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
                                    "tools", "tools.*"]),
    dependency_links=DEPENDENCY_LINKS,
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
