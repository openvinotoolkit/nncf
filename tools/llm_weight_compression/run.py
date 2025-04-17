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

import argparse
import itertools
import json
import shutil
import subprocess
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from optimum.intel import OVModelForCausalLM
from tabulate import tabulate
from transformers import AutoTokenizer

LM_EVAL_RESULTS_FILENAME = "lm_eval_results.json"
OPTIMUM_CLI_PARAMS_FILENAME = "optimum_cli_params.json"
WWB_METRICS_FILENAME = "metrics.csv"


class CompressBackendType(Enum):
    OPTIMUM_CLI = "optimum_cli"
    NNCF = "nncf"


def export_base_model(model_id: str, base_model_dir: Path) -> None:
    """
    Exports a base openvino model into the following folder structure

        {ROOT_DIR}
        |-- {encoded model ID}
            |-- fp32
                |-- model
                    |-- openvino_model.xml
                    |-- openvino_model.bin
                    |-- ...

    :param model_id: A model ID of a model hosted on the [Hub](https://huggingface.co/models).
    :param base_model_dir: A directory where the model should be saved.
    """
    model = OVModelForCausalLM.from_pretrained(
        model_id=model_id, export=True, load_in_8bit=False, load_in_4bit=False, compile=False, trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model.save_pretrained(base_model_dir.joinpath("model"))
    tokenizer.save_pretrained(base_model_dir.joinpath("model"))


def dump_all_packages(output_file: str) -> None:
    """
    Generates a list of all installed Python packages and save it to a file.

    :param output_file: The path to the file where the package list
        should be saved.
    """
    with open(output_file, "w") as f:
        subprocess.run(["pip", "freeze"], stdout=f)


def load_json(path: str):
    with open(path, encoding="utf8") as f:
        return json.load(f)


def save_json(data, path: str, indent: int = 4):
    with open(path, "w", encoding="utf8") as outfile:
        json.dump(data, outfile, indent=indent)


# ------------------------------ Params Grid ------------------------------


class Params:
    """ """

    def get_key(self) -> str:
        """ """
        raise NotImplementedError

    def save_to_json(self, path: str) -> None:
        """
        :param path:
        """
        raise NotImplementedError


@dataclass
class OptimumCLIParams(Params):
    # -------------------------------------- #
    task: Optional[str] = None
    trust_remote_code: Optional[bool] = True
    weight_format: Optional[str] = "fp32"
    # -------------------------------------- #
    ratio: Optional[float] = None
    sym: bool = False
    group_size: Optional[int] = None
    backup_precision: Optional[str] = None
    dataset: Optional[str] = None
    all_layers: bool = False
    # -------------------------------------- #
    awq: bool = False
    scale_estimation: bool = False
    gptq: bool = False
    lora_correction: bool = False

    def get_key(self) -> str:
        # Skipped: task, trust_remote_code
        key_items = []
        key_items.append(f"{self.weight_format}")
        if self.sym:
            key_items.append("sym")
        if self.ratio is not None:
            key_items.append(f"r{self.ratio}")
        if self.group_size is not None:
            key_items.append(f"gs{self.group_size}")
        if self.backup_precision is not None:
            key_items.append(f"{self.backup_precision}")
        if self.dataset:
            key_items.append(f"{self.dataset}")

        for field_name in ["all_layers", "awq", "scale_estimation", "gptq", "lora_correction"]:
            if getattr(self, field_name):
                key_items.append(field_name)

        return "_".join(key_items)

    def save_to_json(self, path: str) -> None:
        data = asdict(self)
        save_json(data, path)


@dataclass
class NNCFAPIParams(Params):
    pass


def optimum_cli_create_params_grid(compression_params: List[Dict[str, List[Any]]]) -> List[OptimumCLIParams]:
    """ """
    params_grid = []
    for p in compression_params:
        params_grid.extend(get_all_param_combinations(p, OptimumCLIParams))
    return params_grid


def nncf_create_params_grid(compression_params: List[Dict[str, List[Any]]]) -> List[NNCFAPIParams]:
    raise NotImplementedError


def visualize_experiments(model_id: str, params_grid: List[Params]):
    """
    :param model_id:
    :param params_grid:
    """
    rows = [[model_id, params.get_key()] for params in params_grid]
    print(f"List of configurations to test out ({len(params_grid)}):")
    print(tabulate(tabular_data=rows, headers=["Model ID", "Experiment"], tablefmt="mixed_grid"))


# ------------------------------ Params Grid ------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="A model ID of a model hosted on the [Hub](https://huggingface.co/models)",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--show-only", action="store_true")
    parser.add_argument("--dump-packages", action="store_true")
    return parser.parse_args()


def encode_model_id(model_id):
    """
    :param model_id:
    """
    # return name.replace("/", "_").replace(".", "_")
    if "/" in model_id:
        model_id = "/".join(model_id.split("/")[1:])
    return model_id.replace("/", "_").replace(".", "_")


def run_command(command: str) -> None:
    print(f"Run command: {command}")
    subprocess.run(command, check=True, shell=True)


def run_optimum_cli(
    model_id: Path,
    output_dir: Path,
    params: OptimumCLIParams,
    log_filename: Optional[str] = None,
) -> None:
    """
    :param model_id:
    :param output_dir:
    :param params:
    :param log_filename:
    """
    cmd_line = "optimum-cli"
    cmd_line += " export openvino"
    cmd_line += f" --model {model_id}"
    if params.task:
        cmd_line += f" --task {params.task}"
    if params.trust_remote_code:
        cmd_line += " --trust-remote-code"
    if params.weight_format:
        cmd_line += f" --weight-format {params.weight_format}"
    if params.ratio:
        cmd_line += f" --ratio {params.ratio}"
    if params.sym:
        cmd_line += " --sym"
    if params.group_size:
        cmd_line += f" --group-size {params.group_size}"
    if params.backup_precision:
        cmd_line += f" --backup-precision {params.backup_precision}"
    if params.dataset:
        cmd_line += f" --dataset {params.dataset}"
    if params.all_layers:
        cmd_line += " --all-layers"
    if params.awq:
        cmd_line += " --awq"
    if params.scale_estimation:
        cmd_line += " --scale-estimation"
    if params.gptq:
        cmd_line += " --gptq"
    if params.lora_correction:
        cmd_line += " --lora-correction"

    # output argument
    cmd_line += f" {output_dir.joinpath('model').as_posix()}"

    if log_filename:
        optimum_cli_log = output_dir.joinpath("optimum_cli_log.txt")
        cmd_line += f" 2>&1 | tee -a {optimum_cli_log.as_posix()}"

    return run_command(cmd_line)


def run_nncf(
    model_id: Path,
    output_dir: Path,
    params: NNCFAPIParams,
    log_filename: Optional[str] = None,
) -> None:
    """
    :param model_id:
    :param output_dir:
    :param params:
    :param log_filename:
    """
    raise NotImplementedError


def get_all_param_combinations(experiment: Dict[str, List[Any]], cls) -> List[Params]:
    keys = experiment.keys()
    values = experiment.values()
    combinations = [cls(**dict(zip(keys, combination))) for combination in itertools.product(*values)]
    return combinations


class EvaluateBackendType(Enum):
    LM_EVAL = "lm_eval"
    WHO_WHAT_BENCHMARK = "who_what_benchmark"


def run_lm_eval(model_dir: Path, evaluation_params: Dict[str, Any]):
    """
    :param model_dir:
    :param evaluation_params:
    """
    cmd_line = "lm_eval"
    cmd_line += " --model openvino"

    tasks_arg = ",".join(evaluation_params["tasks"])
    cmd_line += f" --tasks {tasks_arg}"

    cmd_line += f" --model_args pretrained={model_dir.joinpath('model').as_posix()}"

    num_fewshot = evaluation_params.get("num_fewshot")
    if num_fewshot:
        cmd_line += f" --num_fewshot {num_fewshot}"

    batch_size = evaluation_params.get("batch_size")
    if batch_size:
        cmd_line += f" --batch_size {batch_size}"

    device = evaluation_params.get("device")
    if device:
        cmd_line += f" --device {device}"

    cmd_line += f" --output_path {model_dir.joinpath(LM_EVAL_RESULTS_FILENAME).as_posix()}"

    limit = evaluation_params.get("limit")
    if limit:
        cmd_line += f" --limit {limit}"

    cmd_line += " --trust_remote_code"

    return run_command(cmd_line)


def run_who_what_benchmark(model_dir: Path, base_model_dir: Path, evaluation_params: Dict[str, Any]):
    if model_dir.resolve() == base_model_dir.resolve():
        return

    language = evaluation_params["language"]
    gt_data_filename = f"gt_{language}.csv"

    cmd_line = "wwb"
    cmd_line += f" --base-model {base_model_dir.joinpath('model')}"
    cmd_line += f" --target-model {model_dir.joinpath('model')}"
    cmd_line += f" --gt-data {base_model_dir.joinpath(gt_data_filename)}"
    cmd_line += f" --model-type {evaluation_params['model_type']}"
    cmd_line += f" --device {evaluation_params['device']}"
    cmd_line += f" --language {language}"
    # cmd_line += " --hf"
    cmd_line += f" --output {model_dir.as_posix()}"

    return run_command(cmd_line)


def evaluate(model_dir: Path, base_model_dir: Path, evaluation_config: Dict[str, Any]):
    """ """
    backend = EvaluateBackendType(evaluation_config["backend"])
    evaluation_params = evaluation_config["params"]

    print(f"Run evaluation ({backend.name}): {model_dir.as_posix()}")

    if backend == EvaluateBackendType.LM_EVAL:
        run_lm_eval(model_dir, evaluation_params)

    if backend == EvaluateBackendType.WHO_WHAT_BENCHMARK:
        run_who_what_benchmark(model_dir, base_model_dir, evaluation_params)


def compress(model_id: str, root_model_dir: Path, compression_config: Dict[str, Any], show_only: bool = False) -> None:
    """
    :param model_id:
    :param root_model_dir:
    :param compression_config:
    """
    backend = CompressBackendType(compression_config["backend"])
    compression_params = compression_config["params"]

    if backend == CompressBackendType.OPTIMUM_CLI:
        grid = optimum_cli_create_params_grid(compression_params)
    elif backend == CompressBackendType.NNCF:
        grid = nncf_create_params_grid(compression_params)

    visualize_experiments(model_id, grid)

    if show_only:
        return

    for params in grid:
        EXPERIMENT_DIR = root_model_dir / params.get_key()
        if EXPERIMENT_DIR.exists():
            shutil.rmtree(EXPERIMENT_DIR)
        EXPERIMENT_DIR.mkdir(exist_ok=True, parents=True)

        print(f"Applying configuration: {params.get_key()}")

        if backend == CompressBackendType.OPTIMUM_CLI:
            params_filename = OPTIMUM_CLI_PARAMS_FILENAME
            run_optimum_cli(model_id, EXPERIMENT_DIR, params)
        elif backend == CompressBackendType.NNCF:
            params_filename = "nncf_params.json"
            run_nncf(model_id, EXPERIMENT_DIR, params)

        # --------- Save params ---------
        print(f"Saving compression parameters: {EXPERIMENT_DIR / params_filename}")
        params.save_to_json(EXPERIMENT_DIR / params_filename)


class ResultsParser:

    @staticmethod
    def parse_lm_eval_metrics(path: Path):

        METRICS = [
            "acc",
            "ppl",
            "word_perplexity",
            "exact_match,strict-match",
            "perplexity",
            "similarity",
            "fdt_norm",
        ]
        METRICS.extend([metric + ",none" for metric in METRICS])

        data = load_json(path)
        limit = data.get("config", {}).get("limit", None)
        results_section = data.get("results")

        results = []
        for task, task_results in results_section.items():
            res = {}
            for metric, value in task_results.items():
                res["task"] = task

                if metric in METRICS:
                    metric = metric.replace(",none", "")
                    res[metric] = value
            res["limit"] = limit
            results.append(res)

        return results

    @staticmethod
    def parse_who_what_benchmark_metrics(path: Path):
        df = pd.read_csv(path)

        val = {}
        for name in df:
            if name in ["similarity", "FDT", "FDT norm", "SDT", "SDT norm"]:
                val[name] = float(df[name][0])

        return val

    @staticmethod
    def parse_optimum_params(path: Path, fields: List[str]):
        data = load_json(path)
        return {field_name: data[field_name] for field_name in fields}

    @staticmethod
    def parse(root_model_dir: Path):
        c = {}  # configuration_key -> {/* data */}

        for model_dir in root_model_dir.iterdir():
            if not model_dir.is_dir():
                continue

            configuration_key = model_dir.name

            c[configuration_key] = {}
            c[configuration_key]["model"] = root_model_dir.name
            c[configuration_key]["configuration"] = configuration_key

            # Parse the `lm_eval_results.json` file
            path = model_dir.joinpath(LM_EVAL_RESULTS_FILENAME)
            if path.exists():
                c[configuration_key]["lm_eval"] = ResultsParser.parse_lm_eval_metrics(path)

            # Parse the WWB metrics file
            path = model_dir.joinpath(WWB_METRICS_FILENAME)
            if path.exists():
                # TODO(andrey-churkin): Find the format specification for the `metrics.csv` file
                c[configuration_key]["who_what_benchmark"] = ResultsParser.parse_who_what_benchmark_metrics(path)

            # Parse the `optimum_cli_params.json` file
            path = model_dir.joinpath(OPTIMUM_CLI_PARAMS_FILENAME)
            if path.exists():
                # TODO(andrey-churkin): Add more fields
                c[configuration_key]["optimum_params"] = ResultsParser.parse_optimum_params(
                    path, ["weight_format", "ratio", "group_size"]
                )

        return c


def save_results(results: Dict[str, Dict[str, Any]], root_path: Path):
# {
#     "int4_r0.2_gs64_auto": {
#         "model": "opt-125m",
#         "configuration": "int4_r0.2_gs64_auto",
#         "lm_eval": [{...}, ...],
#         "optimum_params": {
#             "weight_format": "int4",
#             "ratio": 0.2,
#             "group_size": 64
#         }
#     },
#     "fp32": {
#         "model": "opt-125m",
#         "configuration": "fp32",
#         "lm_eval": [{...}, ...],
#     },
#      ...
# }
    rows: List[Dict[str, Any]] = []
    for val in results.values():
        row = {
            "model": val["model"],
            "configuration": val["configuration"],
        }

        # Add optimum params
        row.update(val.get("optimum_params", {}))
        # Add who_what_benchmark results
        row.update(val.get("who_what_benchmark", {}))

        # Add lm_eval results
        lm_eval = val.get("lm_eval", [])
        if lm_eval:
            new_rows = []
            for dct in lm_eval:
                new_row = row.copy()
                new_row.update(dct)
                new_rows.append(new_row)
        else:
            new_rows = [row]

        rows.extend(new_rows)

    pd.set_option("display.precision", 2)
    df = pd.DataFrame(rows)
    df.to_csv(root_path / "raw_results.csv")

    dump_to_excel(df, root_path / "results.xlsx")


def dump_to_excel(df, output_path: Path):
    # to have all columns, not only pivot's values, but also index one.
    print(df.columns)

    writer = pd.ExcelWriter(output_path, engine="xlsxwriter")
    df.to_excel(writer, sheet_name="all", index=False)
    # (max_row, max_col) = df.shape
    workbook = writer.book
    worksheet = writer.sheets["all"]

    format1 = workbook.add_format({"num_format": "#,##0.00"})
    worksheet.set_column("A:X", 18, format1)
    col_names = [{"header": col_name} for col_name in df.columns]
    worksheet.add_table(
        0,
        0,
        df.shape[0],
        df.shape[1] - 1,
        {
            "columns": col_names,
            # 'style' = option Format as table value and is case sensitive
            # (look at the exact name into Excel)
            "style": None,
        },
    )
    worksheet.autofit()
    workbook.close()
    print("Path to parsed results: ", output_path.resolve())


def main():
    args = parse_args()

    ROOT_DIR = Path(args.root_dir)
    ROOT_MODEL_DIR = ROOT_DIR / encode_model_id(args.model_id)

    # --------- Export base model ---------
    BASE_MODEL_DIR = ROOT_MODEL_DIR / "fp32"
    if BASE_MODEL_DIR.exists():
        shutil.rmtree(BASE_MODEL_DIR)
    BASE_MODEL_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Saving a base model: {BASE_MODEL_DIR}")
    export_base_model(args.model_id, BASE_MODEL_DIR)

    config = load_json(args.config)

    # --------- Compress ---------
    compression_config = config["compression"]
    compress(args.model_id, ROOT_MODEL_DIR, compression_config, show_only=args.show_only)

    if args.show_only:
        return

    # --------- Evaluate ---------
    evaluation_config = config["evaluation"]
    for model_dir in ROOT_MODEL_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        try:
            evaluate(model_dir, BASE_MODEL_DIR, evaluation_config)
        except Exception as e:
            print(e)

    # --------- Save extra info ---------
    if args.dump_packages:
        dump_all_packages(ROOT_MODEL_DIR / "versions.txt")

    # --------- Parse results ---------
    results = ResultsParser.parse(ROOT_MODEL_DIR)

    # --------- Save results ---------
    save_results(results, ROOT_MODEL_DIR)


if __name__ == "__main__":
    main()
