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

import torch


def main(argv):
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input-folder", help="Path to directory with given checkpoints to modify", required=True
    )
    parser.add_argument("-o", "--output-folder", help="Path to directory to save modified checkpoints", required=True)
    args = parser.parse_args(args=argv)

    src_dir = args.input_folder
    dst_dir = args.output_folder
    if not exists(dst_dir):
        makedirs(dst_dir)

    pth_files = [
        (join(src_dir, f), join(dst_dir, f))
        for f in listdir(src_dir)
        if isfile(join(src_dir, f)) and (".pth" in f or ".sd" in f)
    ]

    for pair in pth_files:
        src_file, dst_file = pair
        sd = pth = torch.load(src_file)
        if "state_dict" in pth:
            sd = pth["state_dict"]

        new_sd = {}
        for k, v in sd.items():
            if "activation_quantizers" in k and "INPUT" not in k:
                key_split = k.split(".")
                key_split[-2] += "|OUTPUT"
                k = ".".join(key_split)
            new_sd[k] = v
        pth["state_dict"] = new_sd
        torch.save(pth, dst_file)


if __name__ == "__main__":
    main(sys.argv[1:])
