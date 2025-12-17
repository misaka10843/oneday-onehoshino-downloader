import logging
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import unicodedata

import threading
from concurrent.futures import ThreadPoolExecutor

from utils import config
from utils.logger import setup_logger

from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import random

DATA_DIR = os.path.join(config.DATA_DIR, "data", "paddle")
os.environ["PADDLE_PDX_CACHE_HOME"] = DATA_DIR
os.environ["PADDLE_OCR_BASE_DIR"] = DATA_DIR

os.environ['GLOG_minloglevel'] = '2'

from paddleocr import PaddleOCR

local_data = threading.local()
init_lock = threading.Lock()
OCR_KWARGS = {}

def get_ocr_instance():
    if not hasattr(local_data, "ocr"):
        with init_lock:
            logging.getLogger("ppocr").setLevel(logging.WARNING)
            local_data.ocr = PaddleOCR(**OCR_KWARGS)
    return local_data.ocr

INPUT_DIR = Path(config.DATA_DIR) / "oneday-onehoshino"
DEST_ROOT = Path(config.DATA_DIR) / "一日一星野"
CHAPTER_DIR = DEST_ROOT / "Chapter"
OTHER_DIR = DEST_ROOT / "Other"
DEBUG_DIR = DEST_ROOT / "Debug"
CHAPTER_DIR.mkdir(parents=True, exist_ok=True)
OTHER_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# 判断一日一星野的 D+{天数}~{页数} 或 D+{天数}(页数)
# 判断一日一星野的 D+{天数}~{页数} 或 D+{天数}(页数)，支持 D- (前日谈)
# Group 1: sign (+ or -)
# Group 2: AID
# Group 3: BID (optional)
PAT_PAREN = re.compile(r"D([+\-])(\d{1,6})[（(](\d{1,4})[)）]")
PAT_TILDE = re.compile(r"D([+\-])(\d{1,6})[~～](\d{1,4})")
PAT_SIMPLE = re.compile(r"D([+\-])(\d{1,6})(?![（(~～])")


def fix_ocr_typos(s: str) -> str:
    """
    修正 OCR 常见错别字，例如 l -> 1, O -> 0
    仅针对数字部分的混淆进行替换，小心误伤 D
    """
    # 简单替换，因为我们预期的是 D+数字 格式
    # D 必须保留，+ - 必须保留
    # 剩下的 l, I, i -> 1; O, o -> 0; S, s -> 5; Z, z -> 2
    # 注意不要把 D 替换了
    if not s:
        return s
    
    # 混淆字符映射
    table = str.maketrans({
        "l": "1", "I": "1", "i": "1",
        "O": "0", "o": "0",
        "S": "5", "s": "5",
        "Z": "2", "z": "2"
    })
    return s.translate(table)


def normalize_token(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)  # 全角转半角
    return s.strip()


def boxes_to_xyxy(box):
    """
    将 rec_boxes 或 rec_polys 的单个框统一为 (xmin, ymin, xmax, ymax)
    """
    if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
        xmin, ymin, xmax, ymax = box
        return float(xmin), float(ymin), float(xmax), float(ymax)
    if isinstance(box, (list, tuple)) and len(box) >= 3 and all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in box):
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
        
    return None


def group_tokens_by_line(texts: List[str], boxes: List) -> List[List[Tuple[str, Tuple[float, float, float, float]]]]:
    """
    基于 bbox 的 y 中心与高度做粗略分行，再按 x 排序
    """
    items = []
    for t, b in zip(texts, boxes):
        t = normalize_token(t)
        if not t:
            continue
        bb = boxes_to_xyxy(b)
        if bb is None:
            continue
        xmin, ymin, xmax, ymax = bb
        ycen = (ymin + ymax) / 2.0
        h = max(1.0, ymax - ymin)
        items.append((t, bb, ycen, h))

    if not items:
        return []

    items.sort(key=lambda x: (x[2], x[1][0]))

    heights = sorted([it[3] for it in items])
    median_h = heights[len(heights) // 2]
    y_tol = max(6.0, 0.6 * median_h)

    lines = []
    cur_line = []
    cur_y_ref = None
    for t, bb, ycen, h in items:
        if cur_line and cur_y_ref is not None and abs(ycen - cur_y_ref) > y_tol:
            # 开新行
            cur_line.sort(key=lambda x: x[1][0])  # 按 x
            lines.append(cur_line)
            cur_line = []
            cur_y_ref = None
        cur_line.append((t, bb))
        if cur_y_ref is None:
            cur_y_ref = ycen
        else:
            cur_y_ref = (cur_y_ref * (len(cur_line) - 1) + ycen) / len(cur_line)

    if cur_line:
        cur_line.sort(key=lambda x: x[1][0])
        lines.append(cur_line)

    return lines


def find_ids_from_tokens(texts: List[str], boxes: List) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    在被拆 token 的情况下提取 AID/BID：
      D+<AID> 或 D-<AID> (前日谈) ...
    返回 (sign, aid, bid) 例如 ('+', '0001', '0001') 或 ('-', '0010', None)
    """
    lines = group_tokens_by_line(texts, boxes)

    def try_match_in_str(s_no_space: str) -> Optional[Tuple[str, str, Optional[str]]]:
        # 预修正
        s_fixed = fix_ocr_typos(s_no_space)
        if "D+" not in s_fixed and "D-" not in s_fixed:
            return None
            
        m = PAT_PAREN.search(s_fixed)
        if m:
            sign, aid, bid = m.group(1), m.group(2), m.group(3)
            return sign, f"{int(aid):04d}", f"{int(bid):04d}"
        m = PAT_TILDE.search(s_fixed)
        if m:
            sign, aid, bid = m.group(1), m.group(2), m.group(3)
            return sign, f"{int(aid):04d}", f"{int(bid):04d}"
        m = PAT_SIMPLE.search(s_fixed)
        if m:
            sign, aid = m.group(1), m.group(2)
            return sign, f"{int(aid):04d}", None
        return None

    # 滑窗拼接，尽量容错
    for line in lines:
        tokens = [t for t, _ in line]
        # 行整体（去空格）先试一次
        joined_all = "".join(tokens)
        joined_all = normalize_token(joined_all).replace(" ", "")
        hit = try_match_in_str(joined_all)
        if hit:
            return hit

        n = len(tokens)
        for i in range(n):
            for j in range(i + 1, min(n, i + 8) + 1):
                window_tokens = tokens[i:j]
                joined = "".join(window_tokens)
                joined = normalize_token(joined).replace(" ", "")
                if ("+" not in joined and "-" not in joined) and not any("+" in tk or "-" in tk for tk in window_tokens):
                    continue
                hit = try_match_in_str(joined)
                if hit:
                    return hit

    # 把所有 token 串起来再试一次
    all_join = "".join([t for t in texts if t]).strip().replace(" ", "")
    all_join = normalize_token(all_join)
    hit = try_match_in_str(all_join)
    if hit:
        return hit

    return None


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        stem = dst.stem
        ext = dst.suffix
        k = 1
        while True:
            cand = dst.with_name(f"{stem}_{k}{ext}")
            if not cand.exists():
                shutil.copy2(src, cand)
                return
            k += 1
    else:
        shutil.copy2(src, dst)


def get_font(size=20):
    fonts = ["msyh.ttc", "simhei.ttf", "arial.ttf"]
    for font_name in fonts:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
    return ImageFont.load_default()


def draw_ocr_debug(img_path: Path, texts: List[str], boxes: List, found_ids: Optional[Tuple[str, Optional[str]]] = None):
    try:
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = get_font(20)
        
        for text, box in zip(texts, boxes):
            points = None
            if isinstance(box, (list, tuple)):
                if len(box) == 4 and isinstance(box[0], (list, tuple)):
                    # Polygon
                    points = [(p[0], p[1]) for p in box]
                elif len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
                    # xyxy
                    xmin, ymin, xmax, ymax = box
                    points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
            
            if points:
                draw.polygon(points, outline="red", width=2)
                # 文字画在左上角
                txt_pos = points[0]
                # 描边效果
                draw.text(txt_pos, text, fill="red", font=font, stroke_width=1, stroke_fill="white")

        if found_ids:
            sign, aid, bid = found_ids
            info = f"MATCH: D{sign}{aid}" + (f", P{bid}" if bid else "")
            # 画在图片顶部
            draw.text((10, 10), info, fill="blue", font=get_font(40), stroke_width=2, stroke_fill="white")
        else:
             draw.text((10, 10), "NO MATCH", fill="red", font=get_font(40), stroke_width=2, stroke_fill="white")

        save_path = DEBUG_DIR / img_path.name
        image.save(save_path)
    except Exception as e:
        logging.error(f"Debug drawing failed for {img_path}: {e}")


def process_one_image(img_path: Path):
    try:
        ocr_instance = get_ocr_instance()
        result_list = ocr_instance.predict(str(img_path))
        all_texts, all_boxes = [], []
        for res in result_list:
            data = res.json
            inner = data.get("res", {})
            texts = inner.get("rec_texts", []) or []
            boxes = inner.get("rec_boxes", []) or inner.get("rec_polys", []) or []
            all_texts.extend(texts)
            all_boxes.extend(boxes)

        ids = find_ids_from_tokens(all_texts, all_boxes)
        
        if config.OCR_DEBUG:
            draw_ocr_debug(img_path, all_texts, all_boxes, ids)

        if not all_texts:
            # 无文本 → Other
            dst = OTHER_DIR / img_path.name
            safe_copy(img_path, dst)
            return False, None

        if ids is None:
            # 未匹配到 → Other
            dst = OTHER_DIR / img_path.name
            safe_copy(img_path, dst)
            return False, None

        sign, aid, bid = ids
        
        prefix = "前日谈-" if sign == "-" else ""
        
        if bid:
            new_name = f"{prefix}C{aid}-P{bid}.jpg"
        else:
            new_name = f"{prefix}C{aid}.jpg"
        dst = CHAPTER_DIR / new_name
        safe_copy(img_path, dst)
        return True, (sign, aid, bid)
    except Exception as e:
        # 出错也放入 Other，保留原名
        dst = OTHER_DIR / img_path.name
        try:
            safe_copy(img_path, dst)
        except Exception:
            pass
        logging.error(f"处理 {img_path} 出错: {e}")
        return False, None


def main():
    imgs = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]
    imgs.sort()
    
    total = len(imgs)
    ok_count = 0
    fail_count = 0
    processed_count = 0
    lock = threading.Lock()

    logging.getLogger("ppocr").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    cpu_count = os.cpu_count() or 4
    # 计算每个 PaddleOCR 实例的线程数，避免过载
    per_process_threads = max(1, cpu_count // config.OCR_THREADS)

    global OCR_KWARGS
    OCR_KWARGS = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "enable_mkldnn": not config.OCR_USE_GPU,
        "device": "gpu" if config.OCR_USE_GPU else "cpu",
        "cpu_threads": per_process_threads
    }

    logging.info(f"开始处理，共 {total} 张图片，使用线程数: {config.OCR_THREADS}，单实例加速线程: {per_process_threads}")

    # 预热线程池
    logging.info("正在预热 OCR 模型...")

    with ThreadPoolExecutor(max_workers=config.OCR_THREADS) as executor:
        futures = [executor.submit(get_ocr_instance) for _ in range(config.OCR_THREADS)]
        for f in futures:
            f.result()
        logging.info("模型预热完成，开始执行识别任务...")

        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        ) as progress:
            
            task_id = progress.add_task("[cyan]OCR Processing...", total=total)
            
            def task_with_progress(p):
                nonlocal ok_count, fail_count
                hit, ids = process_one_image(p)
                with lock:
                    if hit:
                        ok_count += 1
                    else:
                        fail_count += 1
                    progress.advance(task_id)
            
            list(executor.map(task_with_progress, imgs))

    logging.info(f"完成，识别成功 {ok_count} 张，未匹配 {fail_count} 张。输出目录：{DEST_ROOT}")


if __name__ == "__main__":
    setup_logger()
    main()
