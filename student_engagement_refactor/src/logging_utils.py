import logging

def get_logger(name=__name__, level="INFO"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ch.setFormatter(logging.Formatter(fmt))
        logger.addHandler(ch)
    logger.setLevel(getattr(logging, level))
    return logger
