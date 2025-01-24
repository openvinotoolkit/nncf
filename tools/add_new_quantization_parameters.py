# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from argparse import ArgumentParser
from os import listdir
from os import makedirs
from os.path import exists
from os.path import isfile
from os.path import join
from shutil import copyfile
from typing import Any, NamedTuple

import torch

from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import SymmetricQuantizer


class ParameterToAdd(NamedTuple):
    name: str
    value: Any


def main(argv):
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input-folder", help="Path to directory with given checkpoints to modify", required=True
    )
    parser.add_argument("-o", "--output-folder", help="Path to directory to save modified checkpoints", required=True)
    parser.add_argument(
        "-b", "--bitwidth", help="Bitwidth to initialize quantizer", required=False, default=8, type=int
    )
    parser.add_argument(
        "-v", "--verbose", help="Print all new names of parameters", required=False, action="store_true"
    )
    args = parser.parse_args(args=argv)

    src_dir = args.input_folder
    dst_dir = args.output_folder
    if not exists(dst_dir):
        makedirs(dst_dir)

    param_list = [
        ParameterToAdd("_num_bits", torch.IntTensor([args.bitwidth])),
        ParameterToAdd("enabled", torch.IntTensor([1])),
    ]

    pth_files = [
        (join(src_dir, f), join(dst_dir, f))
        for f in listdir(src_dir)
        if isfile(join(src_dir, f)) and (".pth" in f or ".sd" in f)
    ]

    files_to_copy = []
    for pair in pth_files:
        src_file, dst_file = pair
        sd = pth = torch.load(src_file)
        if "state_dict" in pth:
            sd = pth["state_dict"]

        hooks = [SymmetricQuantizer.SCALE_PARAM_NAME, AsymmetricQuantizer.INPUT_LOW_PARAM_NAME]
        new_keys = {}
        for new_parameter in param_list:
            old_keys = list(sd.keys())
            for k in sd:
                for h in hooks:
                    new_key = k.replace(h, new_parameter.name)
                    if ("." + h in k) and ("." + new_parameter.name not in k) and (new_key not in old_keys):
                        new_keys[new_key] = new_parameter.value
        if new_keys:
            print(f"\nAdding #{len(new_keys)} of new keys")
            if args.verbose:
                print("New keys:", new_keys, sep="\n")
            for new_key, value in new_keys.items():
                sd[new_key] = value
            pth["state_dict"] = sd
            torch.save(pth, dst_file)
        else:
            files_to_copy.append(pair)

    for src_file, dst_file in files_to_copy:
        print("\nCopying {}".format(dst_file))
        copyfile(src_file, dst_file)


if __name__ == "__main__":
    main(sys.argv[1:])
