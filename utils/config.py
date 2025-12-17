import os
from pathlib import Path

from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv(verbose=True)

CURRENT_FILE = Path(__file__).resolve()
APP_ROOT = CURRENT_FILE.parent.parent

DATA_DIR = os.environ.get("DATA_DIR") or os.path.join(APP_ROOT, "data")
COOKIES = os.environ.get("COOKIES")
QUNID = os.environ.get("QUNID")
ALBUMID = os.environ.get("ALBUMID")
QQNUM = os.environ.get("QQNUM")
TK = os.environ.get("TK")
ALBUM_NAME = os.environ.get("ALBUM_NAME")
HOSHINO = os.environ.get("HOSHINO") or False
OCR_DEBUG = os.environ.get("OCR_DEBUG", "false").lower() == "true"
OCR_THREADS = int(os.environ.get("OCR_THREADS", 4))