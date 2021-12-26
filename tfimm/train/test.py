"""Temporary file to illustrate argument parsing"""
from dataclasses import MISSING
from pprint import pprint

# This needs to be imported, even though we don't use anything from here, so types
# are correctly registered.
import tfimm.train.train  # noqa: F401
from tfimm.train.config import parse_args


def main():
    # --- Example 1 ---
    # We specify only that we expect a class for a trainer object and implicitly all
    # parameters for that particular class
    cfg = {"trainer_class": MISSING}

    cmdline_args = [
        "--trainer_class=BasicTrainer",
        "--trainer.nb_epochs=10",
        "--trainer.input_shape=(224,224)",
        "--trainer.skip_first_val=True",  # Note, overrides default
    ]
    cfg = parse_args(cfg, cmdline_args)
    # Note that we have used default values from the dataclass for all parameters that
    # were missing.
    pprint(cfg)

    # # --- Example 2 ---
    # # Let's use a config file. First we create the config file
    # yaml_cfg = {"trainer": {"validation_every_it": 100}}
    # dump_config(yaml_cfg, "/tmp/cfg.yaml")
    #
    # # Next we specify that we want to look for a config file
    # cfg = {"trainer_class": MISSING, "cfg_file": MISSING}
    # # We add the config file saved above to the command line parameters
    # cmdline_args += ["--cfg_file=/tmp/cfg.yaml"]
    # cfg = parse_args(cfg, cmdline_args)
    # # Note that `validation_every_it` has been set from the config file.
    # pprint(cfg)


if __name__ == "__main__":
    main()
