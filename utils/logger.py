import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

from rich.logging import RichHandler

from utils import config

LOG_FILE_NAME = datetime.now().strftime("%Y%m%d") + ".log"
LOGGER_FILE = os.path.join(config.DATA_DIR, "logs", LOG_FILE_NAME)

# 日志格式
DATE_FORMAT = "%d-%b-%y %H:%M:%S"
LOGGER_FORMAT = "%(levelname)s: %(asctime)s \t%(message)s"


@dataclass
class LoggerConfig:
    handlers: list
    format: str
    date_format: str
    logger_file: str
    level: int = logging.INFO


@lru_cache
def get_logger_config():
    if not os.path.exists(os.path.join(config.DATA_DIR, "logs")):
        os.makedirs(os.path.join(config.DATA_DIR, "logs"), exist_ok=True)
    file_handler = logging.FileHandler(LOGGER_FILE, encoding='utf-8')
    formatter = logging.Formatter(LOGGER_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    rich_handler = RichHandler(rich_tracebacks=True)

    return LoggerConfig(
        handlers=[file_handler, rich_handler],
        format=LOGGER_FORMAT,
        date_format=DATE_FORMAT,
        logger_file=LOGGER_FILE,
        level=logging.DEBUG,
    )


def setup_logger():
    config_values = get_logger_config()
    logging.basicConfig(
        level=config_values.level,
        format=config_values.format,
        datefmt=config_values.date_format,
        handlers=config_values.handlers,
        force=True
    )

    for name in logging.root.manager.loggerDict.keys():
        log = logging.getLogger(name)
        log.handlers = config_values.handlers
        log.setLevel(config_values.level)
        log.propagate = False

    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("apscheduler.scheduler").setLevel(logging.WARNING)
    logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)

    if config_values.logger_file:
        logging.getLogger().info(f"日志文件路径: {config_values.logger_file}")
