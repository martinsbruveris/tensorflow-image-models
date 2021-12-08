"""
Script to convert models from timm to TF format

Copyright 2021 Martins Bruveris
"""
from pathlib import Path

import click

import tfimm


@click.command()
@click.option("--model", help="Which model to convert. Can be comma-separated list")
@click.option("--output-dir", help="Where to save converted model")
def main(model, output_dir):
    model_names = model.split(",")
    output_dir = Path(output_dir)

    for model_name in model_names:
        print(f"Model {model_name}: ", end="")
        model = tfimm.create_model(model_name)
        output_path = output_dir / model_name
        model.save(output_path)
        print("saved.")


if __name__ == "__main__":
    main()
