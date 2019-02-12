import sys
import logging


def get_logger(level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    return logger


def get_stdout_handler(level=logging.INFO, print_format='%(message)s'):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(print_format)
    handler.setFormatter(formatter)
    return handler
