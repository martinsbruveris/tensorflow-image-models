import logging


def setup_logging():
    """
    Modifies the global logger to print a nicer message.
    """
    logging.basicConfig(level=logging.INFO)
    fmt = logging.Formatter("%(asctime)s: [%(levelname)8s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # We want to change the format of the root logger, which is the first one
    # in the logger.handlers list. A bit hacky, but there we go.
    root = logger.handlers[0]
    root.setFormatter(fmt)
