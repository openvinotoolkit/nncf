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
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

WIKI_PPL = "wikitext, word perplexity"
LAMBADA_ACC = "lambada-openai, acc"
LAMBADA_PPL = "lambada-openai, perplexity"
WWB_SIM = "WWB, similarity"
MODEL_SIZE = "model size, Gb"
INT4_RATIO = "%int4"
INT8_RATIO = "%int8"
MODE = "mode"
LORA_RANK = "lora rank"
PLOT_NAME = "plot name"


EXPECTED_COLUMNS = [
    MODE,
    INT4_RATIO,
    INT8_RATIO,
    LORA_RANK,
    PLOT_NAME,
    MODEL_SIZE,
    WIKI_PPL,
    LAMBADA_ACC,
    LAMBADA_PPL,
    WWB_SIM,
]

GPTQ = "gptq"
INT4 = "int4"
FP32 = "fp32"
INT8 = "int8"

EXPECTED_IN_MODE_COLUMN = [GPTQ, INT4, FP32]  # , INT8]


COMPRESSION_RATE = "compression rate"
AVG_REL_ERROR = "average relative error"


def check_format(df):
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise RuntimeError(f"The following columns are missing: {missing_columns}")

    missing_in_mode = [item for item in EXPECTED_IN_MODE_COLUMN if not any(df[MODE].str.contains(item))]
    if missing_in_mode:
        raise RuntimeError(
            f"The `{MODE}` column must have at least one entry that includes the following words: {missing_in_mode}"
        )


def add_relative_metrics(df):
    df.sort_values(by=[MODEL_SIZE], ascending=False, inplace=True)

    df[AVG_REL_ERROR] = (
        (df[LAMBADA_ACC].iloc[0] - df[LAMBADA_ACC]) / df[LAMBADA_ACC].iloc[0]
        + (df[LAMBADA_PPL] - df[LAMBADA_PPL].iloc[0]) / df[LAMBADA_PPL].iloc[0]
        + (1 - df[WWB_SIM])
        + (df[WIKI_PPL] - df[WIKI_PPL].iloc[0]) / df[WIKI_PPL].iloc[0]
    ) / 4
    df[COMPRESSION_RATE] = df[MODEL_SIZE].iloc[0] / df[MODEL_SIZE]
    df.sort_values(by=[AVG_REL_ERROR], inplace=True)
    return df


def to_markdown(df_, output_file):
    md_columns = [
        MODE,
        INT4_RATIO,
        INT8_RATIO,
        LORA_RANK,
        AVG_REL_ERROR,
        COMPRESSION_RATE,
    ]
    df = df_[md_columns].fillna("")
    prct_fmt = lambda x: f"{x * 100:.0f}%"
    err_fmt = lambda x: f"{x * 100:.1f}%"
    rate_fmt = lambda x: f"{x:.1f}x"
    df[INT4_RATIO] = df[INT4_RATIO].apply(prct_fmt)
    df[INT8_RATIO] = df[INT8_RATIO].apply(prct_fmt)
    df[COMPRESSION_RATE] = df[COMPRESSION_RATE].apply(rate_fmt)
    df[AVG_REL_ERROR] = df[AVG_REL_ERROR].apply(err_fmt)

    def wrap_text(text):
        return "<br>".join(text.split(" "))

    wrapped_headers = [wrap_text(header) for header in df.columns]

    markdown_table = tabulate(df, headers=wrapped_headers, tablefmt="pipe", showindex=False)
    with open(output_file, "w") as f:
        f.write(markdown_table)


def to_plot(df, output_file):
    df[AVG_REL_ERROR] = df[AVG_REL_ERROR] * 100
    lora = df[LORA_RANK].notnull()
    data_free = df[MODE].isin([INT4])
    gptq = df[MODE].str.contains(GPTQ)
    mixed = ~(gptq | lora | data_free | df[MODE].isin([FP32, INT8]))

    BASELINE_ALGO = df[mixed][MODE].iloc[0].split(f"{INT4} + ")[1]
    MODEL_NAME = df[df.columns[0]].iloc[0]

    colors = ["r", "green", "gray", "b"]
    labels = [
        f"100% int4, lora correction + {BASELINE_ALGO}",
        f"100% int4, gptq + {BASELINE_ALGO}",
        "100% int4, data-free",
        f"mixed-precision, {BASELINE_ALGO}",
    ]
    locs = [(10, -10), (10, 2), (0, -10), (10, -10)]
    dfs = list(map(lambda x: df[x], [lora, gptq, data_free, mixed]))

    for color, label, data in zip(colors, labels, dfs):
        plt.plot(data[AVG_REL_ERROR], data[COMPRESSION_RATE], "o-", color=color, label=label)

    for data, loc in zip(dfs, locs):
        for i, row in data.iterrows():
            plt.annotate(
                row[PLOT_NAME],
                (row[AVG_REL_ERROR], row[COMPRESSION_RATE]),
                textcoords="offset points",
                xytext=loc,
                ha="center",
            )

    plt.xlabel("Relative error to fp32 model on 3 tasks, %")
    plt.ylabel("Compression rate relative to fp32 model")
    plt.title(f"Footprint/Accuracy tradeoff for {MODEL_NAME}")
    plt.legend()

    print("Saving image to :", output_file)
    plt.savefig(output_file)
    plt.close()


def visualize(input_file: str, output_dir: str):
    input_file = Path(input_file)
    if not output_dir:
        output_dir = input_file.parent
    output_dir = Path(output_dir)

    df = pd.read_csv(input_file)
    check_format(df)
    df = add_relative_metrics(df)

    output_file = output_dir / (input_file.stem + ".md")
    to_markdown(df, output_file)

    to_plot(df, output_file.with_suffix(".png"))


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_file", help="Input .csv file", default="phi3_asym.csv")
    parser.add_argument("-o", "--output_dir", help="Directory for output files (.csv, .md and .png)", default="")
    args = parser.parse_args(args=argv)

    visualize(args.input_file, args.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
