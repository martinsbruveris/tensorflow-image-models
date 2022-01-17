# These imports have to be absolute because this file is being executed as a script
import tfimm.train.datasets  # noqa: F401
from tfimm.train import run, setup_logging


def main():
    run(cfg={}, parse_args=True)


if __name__ == "__main__":
    setup_logging()
    main()
