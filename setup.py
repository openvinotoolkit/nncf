# Copyright (c) 2023 Intel Corporation
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
# the same behaviour:
# +-------------------------------------+---------------------------------------------+
# |           Old command               |                New command                  |
# +-------------------------------------+---------------------------------------------+
# | python setup.py install             | pip install .                               |
# | python setup.py develop             | pip install -e .                            |
# | python setup.py develop --*arg*     | pip install --install-option="--*arg*" -e  .| <-- removed since pip 23.x
# | python setup.py sdist               | python -m build -s                          | <-- using the "build" package
# | python setup.py bdist_wheel         | python -m build -w                          | <-- pypi.org/project/build/
# | python setup.py bdist_wheel --*arg* | python -m build -w -C--global-option=--*arg*|
# +-------------------------------------+---------------------------------------------+
#
# The majority of the usual setup.py-related metadata is now in the pyproject.toml,
# but the PSF/PyPA/whoever still can't tackle the simple problem of allowing to specify custom, dynamic versions
# via pyproject.toml only.
# Obscure PR discussions such as https://github.com/pypa/setuptools/pull/3885
# reveal that setup.py can actually still stay, and shall be called even considering the pyproject.toml, so
# it is currently to be used solely for setting version based on the commit SHA
# for repo-based installs.


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


setup(version=find_version(os.path.join(here, "nncf/version.py")))

path_to_ninja = glob.glob(str(sysconfig.get_paths()["purelib"] + "/ninja*/ninja/data/bin/"))
if path_to_ninja:
    path_to_ninja = str(path_to_ninja[0] + "ninja")
    if not os.access(path_to_ninja, os.X_OK):
        st = os.stat(path_to_ninja)
        os.chmod(path_to_ninja, st.st_mode | stat.S_IEXEC)
