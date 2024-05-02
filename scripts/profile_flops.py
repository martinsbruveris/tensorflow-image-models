"""
Script to count FLOPS for each model.

Copyright 2021 Martins Bruveris
"""
from pathlib import Path

import click
import pandas as pd
import tensorflow as tf
import tf_keras

import tfimm
from tfimm.utils import to_2tuple
from tfimm.utils.flops import get_flops, get_parameters


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
                "flops",
                "parameters",
            ]
        )
        results_df.index.name = "model"

    model_names = [name for name in model_names if name not in results_df.index]

    for model_name in model_names:
        print(f"Model: {model_name}. ", end="")
        try:
            tf_keras.backend.clear_session()
            input_size = to_2tuple(input_size) if input_size is not None else input_size
            model = tfimm.create_model(
                model_name, input_size=input_size, nb_classes=nb_classes
            )
            input_shape = (1, *model.cfg.input_size, model.cfg.in_channels)
            input_size = model.cfg.input_size[0]
            flops = get_flops(model, input_shape=input_shape)
            parameters = get_parameters(model)
        except tf.errors.InvalidArgumentError:
            input_size = 0
            flops = 0
            parameters = 0

        print(f"FLOPS: {flops}.")
        results_df.loc[model_name, "image_size"] = input_size
        results_df.loc[model_name, "flops"] = flops
        results_df.loc[model_name, "parameters"] = parameters
        results_df.to_csv(results_file)

    # Some final massaging of results
    results_df.sort_index(inplace=True)
    results_df["image_size"] = results_df["image_size"].astype(int)
    results_df["flops"] = results_df["flops"].astype(int)
    results_df["parameters"] = results_df["parameters"].astype(int)
    results_df.to_csv(results_file)


if __name__ == "__main__":
    main()
