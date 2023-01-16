"""
Copyright (c) 2023 Intel Corporation
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


#pylint:skip-file
import json
import sys
from collections import OrderedDict
from typing import List, Optional

#pylint:disable=import-error
from mdutils import MdUtils
import argparse

from tests.torch.test_sota_checkpoints import DIFF_FP32_MAX_GLOBAL
from tests.torch.test_sota_checkpoints import DIFF_FP32_MIN_GLOBAL

BASE_PYTORCH_CHECKPOINT_URL = 'https://storage.openvinotoolkit.org/repositories/nncf/models/v1.7.0/'

SAMPLE_TYPE_TO_SAMPLE_DISPLAY_NAME_DICT = {
    'classification': 'Classification',
    'semantic_segmentation': 'Semantic segmentation',
    'object_detection': 'Object detection'
}


def get_fp32_and_compressed_metrics(model_name_to_metric_dict,
                                    model_name,
                                    reference,
                                    target):
    if reference is None:
        fp32_ref_metric = float(target)
        compressed_metric = None
    else:
        fp32_ref_metric = float(model_name_to_metric_dict[reference])
        compressed_metric = float(model_name_to_metric_dict[model_name])
    return fp32_ref_metric, compressed_metric


# Not perfect - better switch to building a complete table and then excluding
# columns, or use a better table manager in the future
def get_header_row(table_format: str, sample_type: str):
    assert table_format in ['overview', 'per_sample'], "Unsupported table format!"

    header = ["PyTorch Model"] if table_format == 'overview' else ["Model"]
    header.append('Compression algorithm')
    header.append('Dataset')
    if sample_type in ['classification', 'semantic_segmentation']:
        header.append('Accuracy (Drop) %')
    elif sample_type in 'object_detection':
        header.append('mAP (drop) %')
    else:
        raise RuntimeError(f'{sample_type} sample type is not supported!')

    if table_format == 'per_sample':
        header.append("NNCF config file")
        header.append("PyTorch checkpoint")
    return header


def build_per_model_row(table_format:str,
                        model_display_name: str,
                        compression_algo_display_name: str,
                        dataset_display_name: str,
                        fp32_metric: float,
                        compressed_metric: float,
                        config_path: str,
                        checkpoint_url: Optional[str]):
    row = [model_display_name, compression_algo_display_name, dataset_display_name]
    if compression_algo_display_name == 'None':
        row.append(f'{fp32_metric:.2f}')
    else:
        drop = fp32_metric - compressed_metric
        row.append(f'{fp32_metric:.2f}({drop:.2f})')
    if table_format == 'per_sample':
        local_config_path = '/'.join(config_path.split('/')[3:])
        config_name = local_config_path.split('/')[-1]
        row.append('[{}]({})'.format(config_name, local_config_path))
        if checkpoint_url is not None:
            row.append('[Link]({})'.format(checkpoint_url))
        else:
            row.append('-')
    return row


def get_results_table_rows(per_sample_config_dict,
                           model_name_to_metric_dict,
                           sample_type,
                           table_format: str) -> List[List[str]]:
    rows = []
    assert table_format in ['overview', 'per_sample'], "Unsupported table format!"

    header = get_header_row(table_format=table_format, sample_type=sample_type)
    rows.append(header)

    for data_name_ in per_sample_config_dict:
        dataset_name = get_display_dataset_name(data_name_)
        model_dicts = per_sample_config_dict[data_name_]
        for model_name in model_dicts:
            conf_file = model_dicts[model_name].get('config', {})
            reference = None
            target = None
            if model_dicts[model_name].get('reference', {}):
                reference = model_dicts[model_name].get('reference', {})
            if model_dicts[model_name].get('target', {}):
                target = model_dicts[model_name]['target']
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

            fp32_ref_metric, compressed_metric = get_fp32_and_compressed_metrics(model_name_to_metric_dict,
                                                                                 model_name,
                                                                                 reference,
                                                                                 target)
            if table_format == 'overview' and compression == 'None':
                continue  # The overview already has baseline results as a separate column
            rows.append(build_per_model_row(table_format,
                                            model_display_name,
                                            compression,
                                            dataset_name,
                                            fp32_ref_metric,
                                            compressed_metric,
                                            conf_file,
                                            checkpoint_link))

    return rows


def update_target_metrics_and_thresholds(config_dict: dict, model_name_to_metric_dict):
    for sample_name in config_dict:
        for dataset_name in config_dict[sample_name]:
            dataset_dict = config_dict[sample_name][dataset_name]
            for model_name in dataset_dict:
                model_dict = config_dict[sample_name][dataset_name][model_name]
                model_dict["target"] = model_name_to_metric_dict[model_name]
                if "reference" not in model_dict:
                    continue
                ref_model_name = model_dict["reference"]
                if ref_model_name not in model_name_to_metric_dict:
                    continue

                ref_model_dict = model_name_to_metric_dict[ref_model_name]

                if "diff_fp32_min" in model_dict:
                    diff_fp32_min_value = model_dict["diff_fp32_min"]
                    actual_diff_fp32 = model_name_to_metric_dict[model_name] - \
                                       model_name_to_metric_dict[ref_model_name]
                    if diff_fp32_min_value < DIFF_FP32_MIN_GLOBAL:  # model has special thresholds larger than global
                        if actual_diff_fp32 > diff_fp32_min_value:  # ...but it actually shows better results
                            if actual_diff_fp32 > DIFF_FP32_MIN_GLOBAL:
                                del model_dict["diff_fp32_min"]  # no "diff_fp32_min" means use global
                            else:
                                model_dict["diff_fp32_min"] = round(actual_diff_fp32 - 0.05)  # tighten the threshold
                    if actual_diff_fp32 < diff_fp32_min_value:
                        print(f"Warning: model {model_name} scores less ({actual_diff_fp32}) "
                              f"than the FP32 min threshold {diff_fp32_min_value}")

                if "diff_fp32_max" in model_dict:
                    diff_fp32_max_value = model_dict["diff_fp32_max"]
                    actual_diff_fp32 = model_name_to_metric_dict[model_name] - \
                                       model_name_to_metric_dict[ref_model_name]
                    if diff_fp32_max_value > DIFF_FP32_MAX_GLOBAL:  # model has special thresholds larger than global
                        if actual_diff_fp32 > diff_fp32_max_value:  # ...but it actually shows better results
                            if actual_diff_fp32 < DIFF_FP32_MAX_GLOBAL:
                                del model_dict["diff_fp32_max"]  # no "diff_fp32_max" means use global
                            else:
                                model_dict["diff_fp32_max"] = round(actual_diff_fp32 + 0.05, 1)  # tighten the threshold
                    if actual_diff_fp32 > diff_fp32_max_value:
                        print(f"Warning: model {model_name} scores more ({actual_diff_fp32}) "
                              f"than the FP32 max threshold {diff_fp32_max_value}")


def get_display_dataset_name(data_name):
    if data_name == 'imagenet':
        dataset_name = 'ImageNet'
    elif data_name == 'camvid':
        dataset_name = 'CamVid'
    elif data_name == 'VOCdevkit':
        dataset_name = 'VOC12+07 train, VOC07 eval'
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


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', '-r', help='A metrics.json file from a latest checkpoint evaluation run')
    parser.add_argument('--config', '-c',
                        help='A .json file with definitions of tested checkpoints (sota_checkpoints_eval.json)')
    parser.add_argument('--output', '-o',
                        help="If specified, will output a config file specified in -c with target metric values updated "
                             "to what was measured in the metrics.json file supplied via -r")
    args = parser.parse_args(args=argv)
    results = args.results
    config = args.config
    output = args.output

    measured_metrics = json.load(open(results, 'r'))
    sota_checkpoints_eval = json.load(open(config), object_pairs_hook=OrderedDict)

    # Output tables for per-sample README files
    for sample_type in sota_checkpoints_eval:
        table_rows = get_results_table_rows(sota_checkpoints_eval[sample_type],
                                            measured_metrics,
                                            sample_type,
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
    mdfile.new_line('### PyTorch models')
    mdfile.new_line()
    for sample_type in sota_checkpoints_eval:
        mdfile.new_line('<a name = "pytorch_object_detection" > </a>')
        mdfile.new_line()
        mdfile.new_line(f'#### {SAMPLE_TYPE_TO_SAMPLE_DISPLAY_NAME_DICT[sample_type]}')
        mdfile.new_line()

        table_rows = get_results_table_rows(sota_checkpoints_eval[sample_type],
                                            measured_metrics,
                                            sample_type,
                                            table_format='overview')
        write_table_to_md_file(mdfile, table_rows)
    mdfile.create_md_file()
    delete_four_head_lines(overview_file_name)

    if args.output is not None:
        update_target_metrics_and_thresholds(sota_checkpoints_eval, measured_metrics)
        with open(output, "w") as write_file:
            json.dump(sota_checkpoints_eval, write_file, indent=4)


if __name__ == '__main__':
    main(sys.argv[1:])
