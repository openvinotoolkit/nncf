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
import fileinput
# This is a helper script which parses a metrics.json file containing currently measured accuracy values for checkpoints
# registered in tests/sota_checkpoints_eval.json, and produces:
# a) a file with exactly the same contents as tests/*backend*/sota_checkpoints_eval.json, but with target accuracy
# scores updated to reflect what was currently measured and reported in metrics.json, and
# b) a set of .MD file snippets containing updated results tables, ready for copy-and-paste into corresponding
#  README.md files (a common frontpage README and sample-specific readmes).
#
# Usage:
# python tools/update_eval_results.py -f torch -r path/to/metrics.json -i
# python tools/update_eval_results.py -f tf -r path/to/metrics.json -i


#pylint:skip-file
import json
import sys
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List, Optional
from typing import Tuple

#pylint:disable=import-error
from mdutils import MdUtils
import argparse

from tests.shared.paths import PROJECT_ROOT
from tests.shared.paths import TEST_ROOT
from tests.shared.metric_thresholds import DIFF_FP32_MAX_GLOBAL
from tests.shared.metric_thresholds import DIFF_FP32_MIN_GLOBAL

BASE_CHECKPOINT_URL = 'https://storage.openvinotoolkit.org/repositories/nncf/models/develop/'

@dataclass
class SampleReadmeSubTableDescriptor:
    anchor: str
    model_regex: Optional[str] = None
    model_names: Optional[List[str]] = None
    models_duplicated_from_main_table: Optional[List[str]] = None

@dataclass
class SampleDescriptor:
    path_to_readme: Path
    result_table_anchor_in_main_readme: str
    sub_tables_in_own_readme: Optional[List[SampleReadmeSubTableDescriptor]] = None

RESULTS_ANCHOR_IN_SAMPLE_OWN_README = '<a name="results"></a>'
TORCH_EXAMPLES_PATH = PROJECT_ROOT / 'examples' / 'torch'
TF_EXAMPLES_PATH = PROJECT_ROOT / 'examples' / 'tensorflow'

TORCH_SAMPLE_TYPE_TO_DESCRIPTOR = {
    'classification': SampleDescriptor(
        path_to_readme=TORCH_EXAMPLES_PATH / 'classification' / 'README.md',
        result_table_anchor_in_main_readme='<a name="pytorch_classification"></a>',
        sub_tables_in_own_readme=[
            SampleReadmeSubTableDescriptor(
                anchor='<a name="binarization"></a>',
                model_names=["resnet18_imagenet",
                             "resnet18_imagenet_binarization_xnor",
                             "resnet18_imagenet_binarization_dorefa"],
                models_duplicated_from_main_table=["resnet18_imagenet"]),
            SampleReadmeSubTableDescriptor(
                anchor='<a name="filter_pruning"></a>',
                model_names=["resnet50_imagenet",
                             "resnet50_imagenet_pruning_geometric_median",
                             "resnet18_imagenet",
                             "resnet18_imagenet_pruning_magnitude",
                             "resnet18_imagenet_pruning_geometric_median",
                             "resnet34_imagenet",
                             "resnet34_imagenet_pruning_geometric_median_kd",
                             "googlenet_imagenet",
                             "googlenet_imagenet_pruning_geometric_median"],
                models_duplicated_from_main_table=["resnet50_imagenet",
                                                   "resnet18_imagenet",
                                                   "resnet34_imagenet",
                                                   "googlenet_imagenet"]),
            # Models below are currently not being measured in E2E runs.
            # SampleReadmeSubTableDescriptor(
            #     anchor='<a name="accuracy_aware"></a>',
            #     model_names=["resnet50_imagenet",
            #                  "resnet50_imagenet_pruning_accuracy_aware",
            #                  "resnet18_imagenet",
            #                  "resnet18_imagenet_pruning_accuracy_aware"],
            #     models_duplicated_from_main_table=["resnet50_imagenet",
            #                                        "resnet18_imagenet"])
        ]),
    'semantic_segmentation': SampleDescriptor(
        path_to_readme=TORCH_EXAMPLES_PATH / 'semantic_segmentation' / 'README.md',
        result_table_anchor_in_main_readme='<a name="pytorch_semantic_segmentation"></a>',
        sub_tables_in_own_readme=[
            SampleReadmeSubTableDescriptor(
                anchor='<a name="filter_pruning"></a>',
                model_names=["unet_mapillary",
                             "unet_mapillary_pruning_geometric_median"],
                models_duplicated_from_main_table=["unet_mapillary"])
        ]),
    'object_detection': SampleDescriptor(
        path_to_readme=TORCH_EXAMPLES_PATH / 'object_detection' / 'README.md',
        result_table_anchor_in_main_readme='<a name="pytorch_object_detection"></a>',
        sub_tables_in_own_readme=[
            SampleReadmeSubTableDescriptor(
                anchor='<a name="filter_pruning"></a>',
                model_names=["ssd300_vgg_voc",
                             "ssd300_vgg_voc_pruning_geometric_median"],
                models_duplicated_from_main_table=["ssd300_vgg_voc"])
        ])
}  # type: Dict[str, SampleDescriptor]


TF_SAMPLE_TYPE_TO_DESCRIPTOR = {
    'classification': SampleDescriptor(
        path_to_readme=TF_EXAMPLES_PATH / 'classification' / 'README.md',
        result_table_anchor_in_main_readme='<a name="tensorflow_classification"></a>',
        sub_tables_in_own_readme=[
            SampleReadmeSubTableDescriptor(
                anchor='<a name="filter_pruning"></a>',
                model_names=[
                    "resnet50_imagenet",
                    "resnet50_imagenet_pruning_geometric_median",
                    "resnet50_imagenet_pruning_geometric_median_int8"],
                models_duplicated_from_main_table=["resnet50_imagenet"]),
        ]),
    'segmentation': SampleDescriptor(
        path_to_readme=TF_EXAMPLES_PATH / 'segmentation' / 'README.md',
        result_table_anchor_in_main_readme='<a name="tensorflow_instance_segmentation"></a>'),
    'object_detection': SampleDescriptor(
        path_to_readme=TF_EXAMPLES_PATH / 'object_detection' / 'README.md',
        result_table_anchor_in_main_readme='<a name="tensorflow_object_detection"></a>',
        sub_tables_in_own_readme=[
            SampleReadmeSubTableDescriptor(
                anchor='<a name="filter_pruning"></a>',
                model_names=["retinanet_coco",
                             "retinanet_coco_pruning_geometric_median",
                             "retinanet_coco_pruning_geometric_median_int8"],
                models_duplicated_from_main_table=["retinanet_coco"])
        ])
}  # type: Dict[str, SampleDescriptor]

def get_fp32_and_compressed_metrics(model_name_to_metric_dict: Dict,
                                    model_name: str,
                                    reference: str,
                                    target: str) -> Tuple[float, Optional[float]]:
    if reference is None:
        fp32_ref_metric = float(target)
        compressed_metric = None
    else:
        fp32_ref_metric = float(model_name_to_metric_dict[reference])
        compressed_metric = float(model_name_to_metric_dict[model_name])
    return fp32_ref_metric, compressed_metric


# Not perfect - better switch to building a complete table and then excluding
# columns, or use a better table manager in the future
def get_header_row(table_format: str, sample_type: str) -> List[str]:
    assert table_format in ['overview', 'per_sample'], "Unsupported table format!"
    header = ["Model", 'Compression algorithm', 'Dataset']
    if sample_type == 'classification':
        header.append('Accuracy (_drop_) %')
    elif sample_type == 'semantic_segmentation':
        header.append('mIoU (_drop_) %')
    elif sample_type == 'segmentation':  # fits for TF MaskRCNN only
        header.append('mAP (_drop_) %')
    elif sample_type == 'object_detection':
        header.append('mAP (_drop_) %')
    else:
        raise RuntimeError(f'{sample_type} sample type is not supported!')

    if table_format == 'per_sample':
        header.append("NNCF config file")
        header.append("Checkpoint")
    return header


def build_per_model_row(table_format: str,
                        model_display_name: str,
                        compression_algo_display_name: str,
                        dataset_display_name: str,
                        fp32_metric: float,
                        compressed_metric: float,
                        config_path: Path,
                        containing_file_path: Path,
                        checkpoint_url: Optional[str]) -> List[str]:
    row = [model_display_name, compression_algo_display_name, dataset_display_name]
    if compression_algo_display_name == 'None':
        row.append(f'{fp32_metric:.2f}')
    else:
        drop = fp32_metric - compressed_metric
        row.append(f'{compressed_metric:.2f} ({drop:.2f})')
    if table_format == 'per_sample':
        containing_file_folder = containing_file_path.parent
        local_config_path = config_path.relative_to(containing_file_folder)
        config_name = local_config_path.name
        row.append(f'[{config_name}]({local_config_path.as_posix()})')
        if checkpoint_url is not None:
            row.append(f'[Link]({checkpoint_url})')
        else:
            row.append('-')
    return row


def get_results_table_rows(per_sample_config_dict: Dict,
                           model_name_to_metric_dict: Dict,
                           sample_type: str,
                           table_format: str,
                           framework: str) -> Dict[str, List[List[str]]]:

    assert table_format in ['overview', 'per_sample'], "Unsupported table format!"
    assert framework in ['tf', 'torch'], f"Unsupported framework: {framework}"
    model_name_vs_row = OrderedDict()

    for data_name_ in per_sample_config_dict:
        dataset_name = get_display_dataset_name(data_name_)
        if framework == 'torch':
            model_dicts = per_sample_config_dict[data_name_]
        else:
            model_dicts = per_sample_config_dict[data_name_]["topologies"]
        for model_name in model_dicts:
            conf_file = model_dicts[model_name].get('config')
            if conf_file is not None:
                conf_file_path = PROJECT_ROOT / conf_file
                if not conf_file_path.exists():
                    print(f"Warning: broken config file link {conf_file}")
            else:
                conf_file_path = None
            reference = None
            target = None
            if model_dicts[model_name].get('reference', {}):
                reference = model_dicts[model_name].get('reference', {})
            if model_dicts[model_name].get('target', {}):
                target = model_dicts[model_name]['target']
            resume = model_dicts[model_name].get('resume')
            weights = model_dicts[model_name].get('weights')  # valid for certain TF models
            model_display_name = model_dicts[model_name].get('model_description')

            if model_dicts[model_name].get('compression_description') is not None:
                compression = model_dicts[model_name].get('compression_description')
            else:
                compression = 'None'

            checkpoint_link = None
            ckpt_url = BASE_CHECKPOINT_URL + ('tensorflow/' if framework == 'tf' else 'torch/')
            if resume is not None or weights is not None:
                checkpoint_link = ckpt_url + model_name

            if checkpoint_link is not None:
                if framework == 'tf':
                    checkpoint_link += '.tar.gz'
                else:
                    checkpoint_link += '.pth'

            fp32_ref_metric, compressed_metric = get_fp32_and_compressed_metrics(model_name_to_metric_dict,
                                                                                 model_name,
                                                                                 reference,
                                                                                 target)
            if table_format == 'overview' and compression == 'None':
                continue  # The overview already has baseline results as a separate column

            sample_desc = get_sample_desc(sample_type, framework)
            readme_path = sample_desc.path_to_readme
            model_name_vs_row[model_name] = build_per_model_row(table_format,
                                                                model_display_name,
                                                                compression,
                                                                dataset_name,
                                                                fp32_ref_metric,
                                                                compressed_metric,
                                                                conf_file_path,
                                                                readme_path,
                                                                checkpoint_link)

    sample_type_desc = get_sample_desc(sample_type, framework)
    header = get_header_row(table_format=table_format, sample_type=sample_type)
    if table_format == 'overview':
        rows = [header, ] + list(model_name_vs_row.values())
        return {sample_type_desc.result_table_anchor_in_main_readme: rows}
    elif table_format == 'per_sample':
        retval = {}
        subtables = sample_type_desc.sub_tables_in_own_readme
        if subtables is not None:
            for subtable_desc in subtables:
                subtable_rows = []
                for model_name in subtable_desc.model_names:
                    if model_name not in model_name_vs_row:
                        print(f"Model {model_name} not found among results when trying to create a subtable.")
                        continue
                    subtable_rows.append(model_name_vs_row[model_name])
                    if model_name not in subtable_desc.models_duplicated_from_main_table:
                        del model_name_vs_row[model_name]
                subtable_rows = [header, ] + subtable_rows
                retval[subtable_desc.anchor] = subtable_rows
        retval[RESULTS_ANCHOR_IN_SAMPLE_OWN_README] = [header, ] + list(model_name_vs_row.values())
        return retval


def update_target_metrics_and_thresholds(config_dict: Dict, model_name_to_metric_dict: Dict,
                                         framework: str) -> Dict:
    for sample_name in config_dict:
        for dataset_name in config_dict[sample_name]:
            if framework == 'torch':
                models_for_dataset_dict = config_dict[sample_name][dataset_name]
            elif framework == 'tf':
                models_for_dataset_dict = config_dict[sample_name][dataset_name]["topologies"]
            for model_name, model_dict in models_for_dataset_dict.items():
                model_dict["target"] = model_name_to_metric_dict[model_name]
                if "reference" not in model_dict:
                    continue
                ref_model_name = model_dict["reference"]
                if ref_model_name not in model_name_to_metric_dict:
                    continue

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
    return config_dict


def get_display_dataset_name(data_name: str) -> str:
    if data_name in ['imagenet', 'imagenet2012']:
        dataset_name = 'ImageNet'
    elif data_name == 'camvid':
        dataset_name = 'CamVid'
    elif data_name == 'VOCdevkit':
        dataset_name = 'VOC12+07 train, VOC07 eval'
    elif data_name == 'coco2017':
        dataset_name = "COCO 2017"
    elif data_name == 'mapillary_vistas':
        dataset_name = "Mapillary"
    elif data_name == 'voc':
        dataset_name = "VOC12+07 train, VOC07 eval"
    else:
        raise RuntimeError(f"Unknown data name: {data_name}")
    return dataset_name


def delete_four_head_lines(fname: str):
    with open(fname) as input_data:
        lines = input_data.readlines()
    with open(fname, 'w') as out:
        out.writelines(lines[4:])


def write_table_to_md_file(md_file: MdUtils, table_rows_: List[List[str]]):
    flat_rows = []
    for row in table_rows_:
        flat_rows += row
    md_file.new_table(columns=len(table_rows_[0]), rows=len(table_rows_), text=flat_rows, text_align='center')


def header_name_to_link(header_name_: str) -> str:
    link = header_name_.lower().replace(' ', '-')
    link = '#' + link
    return link


def update_table_inplace(table_content: str, target_file: Path, anchor: str):
    with open(target_file, encoding='utf-8', mode='r') as f:
        old_lines = f.readlines()
        for idx, line in enumerate(old_lines):
            anchor_line = idx
            if line == anchor + '\n':
                break
        else:
            raise RuntimeError(f"Anchor {anchor} not found in target file {target_file}")

        for idx, line in enumerate(old_lines[anchor_line:]):
            table_start_line = anchor_line + idx
            if line.startswith('|'):
                break
        else:
            raise RuntimeError(f"Could not find an MD table to update at anchor {anchor} in {target_file}")

        for idx, line in enumerate(old_lines[table_start_line:]):
            table_end_line = table_start_line + idx
            if not line.startswith('|'):
                break

    content = ''.join(old_lines[:table_start_line]) + table_content + ''.join(old_lines[table_end_line:])
    with open(target_file, encoding='utf-8', mode='w') as f:
        f.write(content)

    print(f"Successfully replaced a table in {target_file} at anchor {anchor}")


def get_per_sample_model_dict(sample_type: str, sota_checkpoints_eval_dict: Dict, framework: str) -> Dict:
    if framework not in ['tf', 'torch']:
        raise RuntimeError(f"Unknown framework type: {framework}")
    if framework == 'torch':
        return sota_checkpoints_eval_dict[sample_type]
    return sota_checkpoints_eval_dict[sample_type]["topologies"]

def filter_tfrecords_only(metrics_dict: Dict[str, float]) -> Dict[str, float]:
    retval = {}
    for model_name, metric_val in metrics_dict.items():
        if model_name.endswith('_tfrecords'):
            new_name = model_name.split('_tfrecords')[0]
            retval[new_name] = metric_val
    return retval


def get_sample_desc(sample_type: str, framework: str) -> SampleDescriptor:
    framework_sample_type_to_desc = TORCH_SAMPLE_TYPE_TO_DESCRIPTOR if framework == 'torch' else TF_SAMPLE_TYPE_TO_DESCRIPTOR
    return framework_sample_type_to_desc[sample_type]


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', '-f', help='The framework for which the eval results are to be updated',
                        choices=['tf', 'torch'], required=True)
    parser.add_argument('--results', '-r', help='A metrics.json file from a latest checkpoint evaluation run',
                        required=True)
    parser.add_argument('--config', '-c',
                        help='A .json file with definitions of tested checkpoints (sota_checkpoints_eval.json)')
    parser.add_argument('--output', '-o',
                        help="If specified, will output a config file specified in -c "
                             "with target metric values updated to what was measured in the metrics.json "
                             "file supplied via -r")
    parser.add_argument('--inplace', '-i',
                        help="If specified, will redact the reference .json file containing target metric levels "
                             "in-place and automatically update the README.md files with new table content, also "
                             "in-place.", action='store_true')
    args = parser.parse_args(args=argv)
    results = args.results
    framework = args.framework
    if framework == 'tf':
        framework = 'tensorflow'

    config = args.config
    if config is None:
        config = TEST_ROOT / framework / 'sota_checkpoints_eval.json'

    output = args.output
    if output is None:
        output = deepcopy(config)

    measured_metrics = json.load(open(results, 'r'))
    if args.framework == 'tf':
        measured_metrics = filter_tfrecords_only(measured_metrics)
    sota_checkpoints_eval = json.load(open(config), object_pairs_hook=OrderedDict)

    # Output tables for per-sample README files
    for sample_type in sota_checkpoints_eval:
        anchor_vs_table_rows = get_results_table_rows(sota_checkpoints_eval[sample_type],
                                                      measured_metrics,
                                                      sample_type,
                                                      table_format='per_sample',
                                                      framework=args.framework)
        file_name = 'results_{}.md'.format(sample_type)
        sample_desc = get_sample_desc(sample_type, args.framework)
        for anchor, table_rows in anchor_vs_table_rows.items():
            mdfile = MdUtils(file_name=file_name)
            write_table_to_md_file(mdfile, table_rows)
            if args.inplace:
                table_str = mdfile.get_md_text()
                table_str = table_str.lstrip('\n')
                update_table_inplace(table_str,
                                     sample_desc.path_to_readme,
                                     anchor)
            else:
                mdfile.create_md_file()
                # Somehow the MDUtils outputs 4 empty lines prior to the actual table in the target file.
                delete_four_head_lines(file_name)

    # Output the overview tables for the top-level README file
    overview_file_name = 'results_overview.md'
    if not args.inplace:
        mdfile = MdUtils(file_name=overview_file_name)

    for sample_type in sota_checkpoints_eval:
        anchor_vs_table_rows = get_results_table_rows(sota_checkpoints_eval[sample_type],
                                                      measured_metrics,
                                                      sample_type,
                                                      table_format='overview',
                                                      framework=args.framework)
        anchor, table_rows = next(iter(anchor_vs_table_rows.items()))
        sample_desc = get_sample_desc(sample_type, args.framework)
        if args.inplace:
            tmp_mdfile = MdUtils(file_name="")
            write_table_to_md_file(tmp_mdfile, table_rows)
            table_str = tmp_mdfile.get_md_text()
            table_str = table_str.lstrip('\n')
            update_table_inplace(table_str,
                                 PROJECT_ROOT / 'README.md',
                                 sample_desc.result_table_anchor_in_main_readme)
        else:
            write_table_to_md_file(mdfile, table_rows)

    if not args.inplace:
        mdfile.create_md_file()
        # Somehow the MDUtils outputs 4 empty lines prior to the actual table in the target file.
        delete_four_head_lines(overview_file_name)

    update_target_metrics_and_thresholds(sota_checkpoints_eval, measured_metrics, args.framework)
    with open(output, "w") as write_file:
        json.dump(sota_checkpoints_eval, write_file, indent=4)


if __name__ == '__main__':
    main(sys.argv[1:])
