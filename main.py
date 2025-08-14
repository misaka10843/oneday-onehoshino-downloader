import logging
import os.path
import re

from functions.downloader import downloader
from functions.get_album import get_album, insert_image
from utils import config
from utils.logger import setup_logger


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', '_', name)

def download():
    album_data = get_album()
    print(f"相册 {album_data['album_id']} {album_data['album_name']} 有 {album_data['total_images']} 张图片")
    print(f"其中 {len(album_data['new_images'])} 张是数据库中未存储的新图片")
    save_path = os.path.join(config.DATA_DIR, album_data['album_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    for index, new_image in enumerate(album_data['new_images']):
        logging.info(f"开始下载{index + 1}张图片")
        if downloader(new_image['url'], os.path.normpath(os.path.join(save_path, f"{sanitize_filename(new_image['id'])}.jpg"))):
            insert_image(album_data['album_id'], new_image['id'])
        else:
            logging.error(f"下载出错：{new_image['url']}")


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    setup_logger()
    download()
