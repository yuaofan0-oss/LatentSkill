import logging


def get_logger(name="Default", filename=None):
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        filename=filename,
    )
    return logging.getLogger(name)
