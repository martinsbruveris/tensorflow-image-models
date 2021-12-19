import logging


def setup_logging(logging_level):
    """
    Creates a logger that logs to stdout. Sets this logger as the global default.
    Logging format is
        2020-12-05 21:44:09,908: Message.

    Returns:
        Doesn't return anything. Modifies global logger.
    """
    logging.basicConfig(level=logging_level)
    fmt = logging.Formatter("%(asctime)s: %(message)s", datefmt="%y-%b-%d %H:%M:%S")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # We want to change the format of the root logger, which is the first one
    # in the logger.handlers list. A bit hacky, but there we go.
    root = logger.handlers[0]
    root.setFormatter(fmt)
