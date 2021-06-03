"""
 Copyright (c) 2021 Intel Corporation
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

from pathlib import Path

TEST_ROOT = Path(__file__).absolute().parents[1]
PROJECT_ROOT = TEST_ROOT.parent.absolute()
EXAMPLES_DIR = PROJECT_ROOT / 'examples'


def get_cli_dict_args(args):
    cli_args = dict()
    for key, val in args.items():
        cli_key = '--{}'.format(str(key))
        cli_args[cli_key] = None
        if val is not None:
            cli_args[cli_key] = str(val)
    return cli_args
