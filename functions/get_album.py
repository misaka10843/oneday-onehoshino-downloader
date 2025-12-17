import logging
import os
import sqlite3

from utils import config
from utils.logger import setup_logger
from utils.request import RequestHandler

# 确保数据库目录存在
os.makedirs(os.path.join(config.DATA_DIR, 'database'), exist_ok=True)
DATABASE_PATH = os.path.join(config.DATA_DIR, 'database', 'album.db')

request = RequestHandler()


def create_table():
    """创建数据库表，如果不存在"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS images
                   (
                       album_id
                       TEXT
                       NOT
                       NULL,
                       image_id
                       TEXT
                       NOT
                       NULL,
                       PRIMARY
                       KEY
                   (
                       album_id,
                       image_id
                   )
                       )
                   ''')
    conn.commit()
    conn.close()


def get_existing_images(album_id):
    """获取指定相册在数据库中已有的图片ID集合"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_id FROM images WHERE album_id = ?", (album_id,))
    existing_images = {row[0] for row in cursor.fetchall()}
    conn.close()
    return existing_images


def insert_image(album_id, image_id):
    """插入单张图片到数据库"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO images (album_id, image_id) VALUES (?, ?)",
            (album_id, image_id)
        )
        conn.commit()
        if cursor.rowcount > 0:
            logging.info(f"已插入图片: 相册ID={album_id}, 图片ID={image_id}")
            conn.close()
            return True
        else:
            logging.info(f"图片已存在，无需插入: 相册ID={album_id}, 图片ID={image_id}")
            conn.close()
            return False
    except sqlite3.Error as e:
        logging.error(f"插入图片出错: {e}")
        raise


def get_album():
    """获取相册list"""
    data = res_album()
    return data


def res_album():
    """获取相册图片列表"""
    url = "https://h5.qzone.qq.com/groupphoto/inqq"
    querystring = {"g_tk": config.TK}
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0 ",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "accept": "application/json, text/javascript, */*; q=0.01",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "sec-ch-ua": "\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
        "x-requested-with": "XMLHttpRequest",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Cookie": config.COOKIES,
    }

    create_table()

    # 获取数据库中已有的图片ID
    existing_images = get_existing_images(config.ALBUMID)

    all_images = []
    new_images = []

    start_num = 0
    while True:
        logging.info(f"正在获取{int(start_num) + 40}条数据")
        payload = f"qunId={config.QUNID}&albumId={config.ALBUMID}&uin={config.QQNUM}&start={start_num}&num=40&getCommentCnt=0&getMemberRole=0&hostUin={config.QQNUM}&getalbum=0&platform=qzone&inCharset=utf-8&outCharset=utf-8&source=qzone&cmd=qunGetPhotoList&qunid={config.QUNID}&albumid={config.ALBUMID}&attach_info=start_count%3D{start_num}"

        response = request.post(url, data=payload, headers=headers, params=querystring)
        if response.status_code != 200:
            raise Exception(f"无法获取到信息，状态码: {response.status_code}")

        data = response.json()

        if data.get('ret') != 0:
            raise Exception(f"API错误: {data}")
        album_name = data['data']['albuminfo']['name']
        # 处理图片列表
        for image in data["data"]["photolist"]:
            image_id = image["sloc"]
            image_url = image["photourl"]["0"]["url"]
            image_info = {'id': image_id, 'url': image_url}
            all_images.append(image_info)

            if image_id not in existing_images:
                new_images.append(image_info)

        if "attach_info" in data["data"] and data["data"]["attach_info"]:
            attach_info = data["data"]["attach_info"]
            start_num = attach_info.split('=')[1]
        else:
            break

    logging.info(f"共获取 {len(all_images)} 张图片, 其中 {len(new_images)} 张为新增")

    # 返回数据库中不存在的图片
    return {
        'album_id': config.ALBUMID,
        'album_name': album_name,
        'total_images': len(all_images),
        'existing_in_db': len(existing_images),
        'new_images': new_images
    }


if __name__ == '__main__':
    setup_logger()
    album_data = get_album()
    print(f"相册 {album_data['album_id']} {album_data['album_name']} 有 {album_data['total_images']} 张图片")
    print(f"其中 {len(album_data['new_images'])} 张是数据库中未存储的新图片")

    # 如果需要，可以将所有新图片插入数据库
    for image in album_data['new_images']:
        print(image)
        insert_image(album_data['album_id'], image['id'])
