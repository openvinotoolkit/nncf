"""
Copyright (c) 2020 Intel Corporation
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

# This is a helper script which parses a metrics.json file containing currently measured accuracy values for checkpoints
# registered in tests/sota_checkpoints_eval.json, and produces:
# a) a file with exactly the same contents as tests/sota_checkpoints_eval.json, but with target accuracy scores
#  updated to reflect what was currently measured and reported in metrics.json, and
# b) a set of .MD file snippets containing updated results tables, ready for copy-and-paste into corresponding
#  README.md files (a common frontpage README and sample-specific readmes).
#
# Usage:
# python update_eval_results.py --results path/to/metrics.json --config path/to/sota_checkpoints_eval.json
# -o path/to/new_config.json

import json
from collections import OrderedDict
from typing import List, Optional

from mdutils import MdUtils
import argparse

BASE_PYTORCH_CHECKPOINT_URL = 'https://storage.openvinotoolkit.org/repositories/nncf/models/v1.5.0'

SAMPLE_TYPE_TO_SAMPLE_DISPLAY_NAME_DICT = {
    'classification': 'Classification',
    'semantic_segmentation': 'Semantic segmentation',
    'object_detection': 'Object detection'
}


def get_fp32_and_compressed_metrics(model_name_to_metric_dict,
                                    model_name,
                                    reference,
                                    table_format):
    if reference is None:
        fp32_ref_metric_str = str(model_name_to_metric_dict[model_name])
        compressed_metric_str = '-' if table_format == 'overview' else fp32_ref_metric_str
    else:
        fp32_ref_metric_str = str(model_name_to_metric_dict[reference])
        compressed_metric_str = str(model_name_to_metric_dict[model_name])
    return fp32_ref_metric_str, compressed_metric_str


# Not perfect - better switch to building a complete table and then excluding
# columns, or use a better table manager in the future
def get_header_row(with_links: bool, with_fp32_baseline: bool):
    header = ["Model", "Compression algorithm", "Dataset"]
    if with_fp32_baseline:
        header.append("PyTorch FP32 baseline")
    header.append("PyTorch compressed accuracy")
    if with_links:
        header.append("Config path")
        header.append("PyTorch checkpoint")
    return header


def build_per_model_row(with_links: bool, with_fp32_baseline: bool,
                        model_display_name: str,
                        compression_algo_display_name: str,
                        dataset_display_name: str,
                        fp32_metric: str,
                        compressed_metric: str,
                        config_path: str,
                        checkpoint_url: Optional[str]):
    row = [model_display_name, compression_algo_display_name, dataset_display_name]
    if with_fp32_baseline:
        row.append(fp32_metric)
    row.append(compressed_metric)
    if with_links:
        row.append(config_path)
        if checkpoint_url is not None:
            row.append('[Link]({})'.format(checkpoint_url))
        else:
            row.append('-')
    return row


def get_results_table_rows(per_sample_config_dict,
                           model_name_to_metric_dict,
                           table_format: str) -> List[List[str]]:
    rows = []
    assert table_format in ['overview', 'per_sample'], "Unsupported table format!"
    if table_format == 'overview':
        with_links = False
        with_fp32_baseline = True
    else:
        with_links = True
        with_fp32_baseline = False

    header = get_header_row(with_links=with_links, with_fp32_baseline=with_fp32_baseline)
    rows.append(header)

    for data_name_ in per_sample_config_dict:
        dataset_name = get_display_dataset_name(data_name_)
        model_dicts = per_sample_config_dict[data_name_]
        for model_name in model_dicts:
            conf_file = model_dicts[model_name].get('config', {})
            reference = None
            if model_dicts[model_name].get('reference', {}):
                reference = model_dicts[model_name].get('reference', {})
            if model_dicts[model_name].get('resume', {}):
                resume = model_dicts[model_name].get('resume', {})
            else:
                resume = None
            model_display_name = model_dicts[model_name].get('model_description')

            if model_dicts[model_name].get('compression_description') is not None:
                compression = model_dicts[model_name].get('compression_description')
            else:
                compression = 'None'

            checkpoint_link = (BASE_PYTORCH_CHECKPOINT_URL + resume) if resume is not None else None

            fp32_ref_metric_str, compressed_metric_str = get_fp32_and_compressed_metrics(model_name_to_metric_dict,
                                                                                         model_name,
                                                                                         reference,
                                                                                         table_format)
            if table_format == 'overview' and compression == 'None':
                continue  # The overview already has baseline results as a separate column
            rows.append(build_per_model_row(with_links, with_fp32_baseline,
                                            model_display_name,
                                            compression,
                                            dataset_name,
                                            fp32_ref_metric_str,
                                            compressed_metric_str,
                                            conf_file,
                                            checkpoint_link))

    return rows


def update_target_metrics(config_dict: dict, model_name_to_metric_dict):
    for sample_name in config_dict:
        for dataset_name in config_dict[sample_name]:
            for model_name in config_dict[sample_name][dataset_name]:
                config_dict[sample_name][dataset_name][model_name] = model_name_to_metric_dict[model_name]


def get_display_dataset_name(data_name):
    if data_name == 'imagenet':
        dataset_name = 'ImageNet'
    elif data_name == 'camvid':
        dataset_name = 'CamVid'
    elif data_name == 'VOCdevkit':
        dataset_name = 'VOC12+07'
    else:
        dataset_name = "Mapillary"
    return dataset_name


def delete_four_head_lines(fname):
    with open(fname) as input_data:
        lines = input_data.readlines()
    with open(fname, 'w') as out:
        out.writelines(lines[4:])


def write_table_to_md_file(md_file: MdUtils, table_rows_: List[List[str]]):
    flat_rows = []
    for row in table_rows_:
        flat_rows += row
    md_file.new_table(columns=len(table_rows_[0]), rows=len(table_rows_), text=flat_rows, text_align='center')


def header_name_to_link(header_name_: str):
    link = header_name_.lower().replace(' ', '-')
    link = '#' + link
    return link

parser = argparse.ArgumentParser()
parser.add_argument('--results', '-r', help='A metrics.json file from a latest checkpoint evaluation run')
parser.add_argument('--config', '-c',
                    help='A .json file with definitions of tested checkpoints (sota_checkpoints_eval.json)')
parser.add_argument('--output', '-o',
                    help="If specified, will output a config file specified in -c with target metric values updated "
                         "to what was measured in the metrics.json file supplied via -r")
args = parser.parse_args()
results = args.results
config = args.config
output = args.output

measured_metrics = json.load(open(results, 'r'))
sota_checkpoints_eval = json.load(open(config), object_pairs_hook=OrderedDict)

# Output tables for per-sample README files
for sample_type in sota_checkpoints_eval:
    table_rows = get_results_table_rows(sota_checkpoints_eval[sample_type],
                                        measured_metrics,
                                        table_format='per_sample')
    file_name = 'results_{}.md'.format(sample_type)
    mdfile = MdUtils(file_name=file_name)
    write_table_to_md_file(mdfile, table_rows)
    mdfile.create_md_file()
    # Somehow the MDUtils outputs 4 empty lines prior to the actual table in the target file.
    delete_four_head_lines(file_name)


# Output the overview table for the top-level README file
overview_file_name = 'results_overview.md'
mdfile = MdUtils(file_name=overview_file_name)

# Compose a mini-TOC
mdfile.new_line("Quick jump to sample type:")
mdfile.new_line("==========================")
for sample_type in sota_checkpoints_eval:
    header_name = SAMPLE_TYPE_TO_SAMPLE_DISPLAY_NAME_DICT[sample_type]
    mdfile.new_line("[{}]({})\n".format(header_name, header_name_to_link(header_name)))
mdfile.new_line()

for sample_type in sota_checkpoints_eval:
    mdfile.new_header(level=4, title=SAMPLE_TYPE_TO_SAMPLE_DISPLAY_NAME_DICT[sample_type])

    table_rows = get_results_table_rows(sota_checkpoints_eval[sample_type],
                                        measured_metrics,
                                        table_format='overview')
    write_table_to_md_file(mdfile, table_rows)
mdfile.create_md_file()
delete_four_head_lines(overview_file_name)

if args.output is not None:
    update_target_metrics(sota_checkpoints_eval, measured_metrics)
    with open(output, "w") as write_file:
        json.dump(sota_checkpoints_eval, write_file, indent=8)
