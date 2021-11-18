from pathlib import Path

import click
import pandas as pd

import tfimm
from tfimm.models import registry
from tfimm.utils.profile import time_inference, time_backprop, find_max_batch_size


@click.command()
@click.option("--results-file", help="Where to save results")
@click.option("--name-filter", type=str, default="", help="Regex to include models")
@click.option("--module", type=str, default="", help="Filter models by module")
@click.option("--exclude-filters", type=str, default="", help="Regex to exclude models")
@click.option("--ignore-results/--no-ignore-results", default=False)
def main(results_file, name_filter, module, exclude_filters, ignore_results):
    model_names = tfimm.list_models(
        name_filter=name_filter,
        module=module,
        exclude_filters=exclude_filters
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
                "infererence_img_per_sec",
                "backprop_img_per_sec",
            ]
        )
        results_df.index.name = "model"

    model_names = [name for name in model_names if name not in results_df.index]

    for model_name in model_names:
        cfg = registry.model_config(model_name)
        results_df.loc[model_name, "image_size"] = cfg.input_size[0]

        batch_size = find_max_batch_size(
            model_name, test_target="inference", verbose=True
        )
        img_per_sec = time_backprop(model_name, batch_size, nb_batches=3)
        results_df.loc[model_name, "inference_batch_size"] = batch_size
        results_df.loc[model_name, "infererence_img_per_sec"] = img_per_sec

        batch_size = find_max_batch_size(
            model_name, test_target="backprop", verbose=True
        )
        img_per_sec = time_inference(model_name, batch_size, nb_batches=3)
        results_df.loc[model_name, "backprop_batch_size"] = batch_size
        results_df.loc[model_name, "backprop_img_per_sec"] = img_per_sec

        results_df.to_csv(results_file)


if __name__ == "__main__":
    main()
