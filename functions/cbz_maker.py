import logging
import os
import re
from pathlib import Path

from cbz import ComicInfo, PageInfo
from cbz.constants import Format, YesNo, Manga, AgeRating
from natsort import natsorted

from utils import config

save_path = os.path.join(config.DATA_DIR,"一日一星野")

def postprocess(series_name: str, chapter_name: str, chapter_filename: str,
                chapter_number: float | int, file_path: str):
    source_dir = Path(file_path)
    if not source_dir.is_dir():
        logging.error(f"无效目录: {file_path}")
        raise ValueError(f"无效目录: {file_path}")

    # 获取并排序文件列表（自然排序）
    paths = natsorted(source_dir.iterdir(), key=lambda x: x.name)
    if not paths:
        logging.error(f"没有在 {file_path} 中找到图片")
        raise RuntimeError(f"没有在 {file_path} 中找到图片")

    pages = [
        PageInfo.load(path=path)
        for path in paths
    ]

    # 构建漫画元数据
    comic = ComicInfo.from_pages(
        pages=pages,
        title=chapter_name,
        series="一日一星野",
        number=chapter_number,
        language_iso='zh',
        format=Format.WEB_COMIC,
        black_white=YesNo.NO,
        manga=Manga.YES,
        age_rating=AgeRating.PENDING
    )

    cbz_base = source_dir.parent / "cbz" / series_name
    cbz_base.mkdir(parents=True, exist_ok=True)

    cbz_path = cbz_base / f"{chapter_filename}.cbz"
    cbz_path.write_bytes(comic.pack())
    logging.info(f"cbz打包成功: {cbz_path}")


def batch_postprocess_from_folder(file_path: str):
    source_dir = Path(file_path)
    if not source_dir.is_dir():
        raise ValueError(f"无效目录: {file_path}")

    chapter_pattern = re.compile(r"^(C\d+)", re.IGNORECASE)

    # 按章节分组文件
    chapters = {}
    for img in source_dir.iterdir():
        if img.is_file() and img.suffix.lower() == ".jpg":
            match = chapter_pattern.match(img.name)
            if match:
                chapter_key = match.group(1)  # 比如 "C0002"
                chapters.setdefault(chapter_key, []).append(img)

    if not chapters:
        logging.error(f"没有在 {file_path} 中找到符合命名规则的图片")
        return

    # 循环处理每个章节
    for chapter_key, files in chapters.items():
        # 自然排序
        files_sorted = natsorted(files, key=lambda x: x.name)

        # 生成章节名/文件名/章节号
        chapter_number = int(chapter_key[1:])  # 去掉 C，转数字
        chapter_name = f"第{chapter_number}话"
        chapter_filename = chapter_key.lower()

        # 创建临时文件夹存储该章节图片
        tmp_chapter_dir = source_dir / f"_tmp_{chapter_key}"
        tmp_chapter_dir.mkdir(exist_ok=True)
        for img in files_sorted:
            img.rename(tmp_chapter_dir / img.name)

        postprocess(
            series_name=source_dir.name,
            chapter_name=chapter_name,
            chapter_filename=chapter_filename,
            chapter_number=chapter_number,
            file_path=str(tmp_chapter_dir)
        )

if __name__ == "__main__":
    batch_postprocess_from_folder(os.path.join(save_path, "Chapter"))