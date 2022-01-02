"""
Script to measure inference speed on CPU.

Copyright 2021 Martins Bruveris
"""
from pathlib import Path

import click
import pandas as pd
import tensorflow as tf

import tfimm
from tfimm.utils.profile import time_model


@click.command()
@click.option("--results-file", help="Where to save results")
@click.option("--name-filter", type=str, default="", help="Regex to include models")
@click.option("--module", type=str, default="", help="Filter models by module")
@click.option("--exclude-filters", type=str, default="", help="Regex to exclude models")
@click.option("--input-size", type=int, default=None, help="Model input resolution")
@click.option("--nb-classes", type=int, default=None, help="Number of classes")
@click.option("--ignore-results/--no-ignore-results", default=False)
def main(
    results_file,
    name_filter,
    module,
    exclude_filters,
    input_size,
    nb_classes,
    ignore_results,
):
    """
    Main function to do the work.

    The parameters `name_filter`, `module` and `exclude_filters` are passed directly to
    `tfimm.list_models` to find which models to profile.

    If `--ignore-results` is set, we ignore any results already existing in the results
    file and rerun profiling for all models. Otherwise (default) we run profiling only
    on models not already in the results file.
    """
    model_names = tfimm.list_models(
        name_filter=name_filter, module=module, exclude_filters=exclude_filters
    )

    results_file = Path(results_file)
    if results_file.exists() and not ignore_results:
        results_df = pd.read_csv(results_file, index_col=0)
    else:
        results_df = pd.DataFrame(
            columns=[
                "inference_time",
                "inference_img_per_sec",
            ]
        )
        results_df.index.name = "model"

    model_names = [name for name in model_names if name not in results_df.index]

    for model_name in model_names:
        print(f"Model: {model_name}. ", end="")

        try:
            img_per_sec = time_model(
                model_name,
                target="inference",
                input_size=input_size,
                nb_classes=nb_classes,
                batch_size=1,
                float_policy="float32",
                nb_batches=5,
            )
            duration = 1.0 / img_per_sec
        except tf.errors.InvalidArgumentError:
            img_per_sec = 0
            duration = 0

        results_df.loc[model_name, "inference_time"] = duration
        results_df.loc[model_name, "inference_img_per_sec"] = img_per_sec
        print(f"Time: {duration:.3f}.")

        results_df.to_csv(results_file)

    # Some final massaging of results
    results_df.sort_index(inplace=True)
    results_df.to_csv(results_file)


if __name__ == "__main__":
    main()
