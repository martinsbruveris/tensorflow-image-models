"""
Script to test for max batch size possible on a GPU for inference and backpropagation
and to measure batch inference and backpropagation throughput in img/sec.

Copyright 2021 Martins Bruveris
"""
from pathlib import Path

import click
import pandas as pd

import tfimm
from tfimm.models import registry
from tfimm.utils.profile import find_max_batch_size, time_backprop, time_inference


@click.command()
@click.option("--results-file", help="Where to save results")
@click.option("--name-filter", type=str, default="", help="Regex to include models")
@click.option("--module", type=str, default="", help="Filter models by module")
@click.option("--exclude-filters", type=str, default="", help="Regex to exclude models")
@click.option("--float-policy", type=str, default="float32", help="mixed precision?")
@click.option("--ignore-results/--no-ignore-results", default=False)
def main(
    results_file, name_filter, module, exclude_filters, float_policy, ignore_results
):
    """
    Main function to do the work.

    The parameters `name_filter`, `module` and `exclude_filters` are passed directly to
    `tfimm.list_models` to find which models to profile.

    The parameter `--float-policy` can be one of "float32" or "mixed_float16".

    If `--ignore-results` is set, we ignore any results already existing in the results
    file and rerun profiling for all models. Otherwise (default) we run profiling only
    on models not already in the results file.
    """
    assert float_policy in {"float32", "mixed_float16"}

    model_names = tfimm.list_models(
        name_filter=name_filter, module=module, exclude_filters=exclude_filters
    )

    results_file = Path(results_file)
    if results_file.exists() and not ignore_results:
        results_df = pd.read_csv(results_file, index_col=0)
    else:
        results_df = pd.DataFrame(
            columns=[
                "image_size",
                "inference_batch_size",
                "backprop_batch_size",
                "inference_img_per_sec",
                "backprop_img_per_sec",
            ]
        )
        results_df.index.name = "model"

    model_names = [name for name in model_names if name not in results_df.index]

    for model_name in model_names:
        print(f"Model: {model_name}")
        cfg = registry.model_config(model_name)
        results_df.loc[model_name, "image_size"] = cfg.input_size[0]

        # Profile inference
        batch_size = find_max_batch_size(
            model_name, test_target="inference", verbose=True
        )
        results_df.loc[model_name, "inference_batch_size"] = batch_size
        if batch_size > 0:
            img_per_sec = time_inference(model_name, batch_size, nb_batches=3)
            results_df.loc[model_name, "inference_img_per_sec"] = round(img_per_sec, 2)
        else:
            img_per_sec = None
        print(f"Inference: {img_per_sec:.3f}img/sec with {batch_size} batch size.")

        # Profile backpropagation
        batch_size = find_max_batch_size(
            model_name, test_target="backprop", verbose=True
        )
        results_df.loc[model_name, "backprop_batch_size"] = batch_size
        if batch_size > 0:
            img_per_sec = time_backprop(model_name, batch_size, nb_batches=3)
            results_df.loc[model_name, "backprop_img_per_sec"] = round(img_per_sec, 2)
        else:
            img_per_sec = None
        print(f"Backprop: {img_per_sec:.3f}img/sec with {batch_size} batch size.")

        results_df.to_csv(results_file)


if __name__ == "__main__":
    main()
