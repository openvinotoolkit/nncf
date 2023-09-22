import pandas as pd

from name_mappings import MODELS, ALL_MODELS, EXCLUDED_MODELS, METRICS, METRICS_PRIORITY, FRAMEWORKS, DATASETS, HEADERS # pylint: disable = no-name-in-module, import-error

# Check that EXCLUDED_MODELS does not intersect with ALL_MODELS
assert len(EXCLUDED_MODELS.intersection(set(ALL_MODELS))) == 0, (
    f"Excluded models set intersects with the models selected for model zoo: "
    f"{EXCLUDED_MODELS.intersection(set(ALL_MODELS))}"
)


# String definitions for convenience
new_line_symbol = "\n"


def enclose(s):
    return f'    "{s}",'


MAX_ACC_DROP = 2  # maximal accuracy drop


# Read table to pandas data frame
df = pd.read_csv("accuracy-snapshot-report.csv")
df = df[
    [
        "topology",
        "dataset",
        "metrictype",
        "metricname",
        "rawmetricvalue",
        "metricreference",
        "sourceframework",
        "metricmeta",
        "metricstatus",
    ]
]

csv_topologies = set(df["topology"].tolist())

# Check that all tracked models are present in the csv table
all_diff_csv = set(ALL_MODELS).difference(csv_topologies)
assert len(all_diff_csv) == 0, f"Not all included models are in the present table: {all_diff_csv}"

# Check that all tracked models are present in table, otherwise there are probably some new ones recently added
csv_diff_all_tracked = csv_topologies.difference(set(ALL_MODELS).union(EXCLUDED_MODELS))
assert len(csv_diff_all_tracked) == 0, (
    f"There are some (new?) models which are not in either included or excluded models: "
    f"{new_line_symbol.join(map(enclose, sorted(csv_diff_all_tracked)))}"
)

# Read topologies.yml from private repo; these models will not be included
with open("topologies_private.yml", "r") as f:
    topologies_private = f.read()

df: pd.DataFrame = df.loc[df["topology"].isin(ALL_MODELS)]
df = df.loc[df["metricstatus"].isin(["passed", "improvement", "downgrade"])]

# Create data frame for every type of model
columns = ["Model", "Framework", "Dataset", "Metric", "Metric Type"]
resulting_dfs = dict([(k, pd.DataFrame(columns=columns)) for k in MODELS.keys()])

private_models = set()
added_frameworks = set()
for topology in ALL_MODELS:
    topology_df: pd.DataFrame = df.loc[df["topology"].isin([topology])]
    if len(topology_df) == 0:
        continue
    if f'name: "{topology}"' in topologies_private:
        private_models.add(topology)
        continue

    # Check if some metric of the model is listed in metric_type_priority
    assert any(
        [k in METRICS for k in topology_df["metricname"]]
    ), f"Some metric from {topology_df['metricname']} is unknown"

    # Sort rows according to metric priority
    topology_df = topology_df.sort_values(
        by="metricname", axis=0, key=lambda series: pd.Series([METRICS_PRIORITY.get(k, 100) for k in series])
    )

    # Select first row
    topology_df = topology_df.iloc[0]
    assert topology_df["metricmeta"] == "higher-better"

    # Find which type of model is this
    resulting_df_key = None
    for k in MODELS.keys():
        if topology in MODELS[k]:
            resulting_df_key = k
            break

    metric_value = topology_df["rawmetricvalue"] * 100
    metric_drop = (topology_df["metricreference"] - topology_df["rawmetricvalue"]) * 100

    # if topology_df["metricstatus"] == "downgrade":
    #     continue
    if metric_drop > MAX_ACC_DROP:
        continue

    if topology_df["sourceframework"] not in FRAMEWORKS:
        continue
    added_frameworks.add(topology_df["sourceframework"])

    pretty_model_name = MODELS[resulting_df_key][topology]
    pretty_dataset_name = DATASETS[topology_df["dataset"]]
    pretty_metric_name = METRICS[topology_df["metricname"]]
    pretty_framework_name = FRAMEWORKS[topology_df["sourceframework"]]

    # Add new data frame row
    resulting_dfs[resulting_df_key].loc[len(resulting_dfs[resulting_df_key])] = {
        "Model": pretty_model_name,
        "Framework": pretty_framework_name,
        "Dataset": pretty_dataset_name,
        "Metric": f"{metric_value:.2f} ({metric_drop:.2f})",
        "Metric Type": pretty_metric_name,
    }

# Drop Framework column if there is only one framework
if len(added_frameworks) == 1:
    for k in resulting_dfs.keys():
        resulting_dfs[k] = resulting_dfs[k].drop(columns=["Framework"])

# Write table
with open("ModelZoo.md", "w") as f:
    for k, df in resulting_dfs.items():
        if len(df) == 0:
            continue
        html_str = df.reset_index(drop=True).to_html(index=False)
        html_str = html_str.replace("text-align: right;", "text-align: center;")
        html_str = html_str.replace(' border="1" class="dataframe"', "")
        f.writelines(HEADERS[k] + "\n\n")
        f.write(html_str.replace("Metric<", "Metric (<em>drop</em>) %<") + "\n\n")


print(f"{len(private_models)} were excluded because belong to private topologies.yml", private_models)
