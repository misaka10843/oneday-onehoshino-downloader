import logging
import os

from utils.request import RequestHandler

request = RequestHandler()


def downloader(url: str, filename: str, overwrite: bool = False) -> bool:
    # 检查文件是否已存在
    if os.path.exists(filename):
        if overwrite:
            logging.warning(f"文件已存在，强制覆盖: {filename}")
        else:
            logging.info(f"文件已存在，跳过下载: {filename}")
            return True

    try:
        # 发起HTTP请求
        response = request.get(url)

        if response is None:
            logging.error(f"无法获取图片响应，URL: {url}")
            return False

        if not response.content:
            logging.error(f"获取到空内容，URL: {url}")
            return False

        # 创建目录路径
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logging.debug(f"创建目录: {directory}")

        # 保存文件
        with open(filename, 'wb') as f:
            f.write(response.content)

        logging.info(f"图片下载成功: {filename}")
        return True

    except IOError as e:
        logging.error(f"文件写入失败: {e}，路径: {filename}")
    except Exception as e:
        logging.error(f"未知错误: {e}，URL: {url}")

    return False
