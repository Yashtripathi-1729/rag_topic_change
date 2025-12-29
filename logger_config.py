# logger_config.py

import logging
import os

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "rag_app.log")

os.makedirs(LOG_DIR, exist_ok=True)


def setup_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # prevent duplicate handlers

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
