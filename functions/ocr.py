import logging
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import unicodedata

from utils import config

DATA_DIR = os.path.join(config.DATA_DIR, "data", "paddle")
os.environ["PADDLE_PDX_CACHE_HOME"] = DATA_DIR
os.environ["PADDLE_OCR_BASE_DIR"] = DATA_DIR

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

INPUT_DIR = config.DATA_DIR / "data" / "oneday-onehoshino"
DEST_ROOT = config.DATA_DIR / "一日一星野"
CHAPTER_DIR = DEST_ROOT / "Chapter"
OTHER_DIR = DEST_ROOT / "Other"
CHAPTER_DIR.mkdir(parents=True, exist_ok=True)
OTHER_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# 判断一日一星野的 D+{天数}~{页数} 或 D+{天数}(页数)
PAT_PAREN = re.compile(r"D\+(\d{1,6})[（(](\d{1,4})[)）]")
PAT_TILDE = re.compile(r"D\+(\d{1,6})[~～](\d{1,4})")
PAT_SIMPLE = re.compile(r"D\+(\d{1,6})(?![（(~～])")


def normalize_token(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)  # 全角转半角
    return s.strip()


def boxes_to_xyxy(box):
    """
    将 rec_boxes 或 rec_polys 的单个框统一为 (xmin, ymin, xmax, ymax)
    - rec_boxes: [xmin, ymin, xmax, ymax]
    - rec_polys: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    try:
        # 可能是 4 个数字
        if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            xmin, ymin, xmax, ymax = box
            return float(xmin), float(ymin), float(xmax), float(ymax)
        # 可能是 4 个点
        if isinstance(box, (list, tuple)) and len(box) == 4 and all(
                isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in box):
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
    except Exception:
        pass
    return None


def group_tokens_by_line(texts: List[str], boxes: List) -> List[List[Tuple[str, Tuple[float, float, float, float]]]]:
    """
    基于 bbox 的 y 中心与高度做粗略分行，再按 x 排序
    返回：每行是 [(text, (xmin,ymin,xmax,ymax)), ...]
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

    # 先按 y 再按 x 排序
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


def find_ids_from_tokens(texts: List[str], boxes: List) -> Optional[Tuple[str, Optional[str]]]:
    """
    在被拆 token 的情况下提取 AID/BID：
      D+<AID> 或 D+<AID>(<BID>) 或 D+<AID>~<BID>
    返回 ('0001', '0001' 或 None)
    """
    lines = group_tokens_by_line(texts, boxes)

    def try_match_in_str(s_no_space: str) -> Optional[Tuple[str, Optional[str]]]:
        if "D+" not in s_no_space:
            return None
        m = PAT_PAREN.search(s_no_space)
        if m:
            aid, bid = m.group(1), m.group(2)
            return f"{int(aid):04d}", f"{int(bid):04d}"
        m = PAT_TILDE.search(s_no_space)
        if m:
            aid, bid = m.group(1), m.group(2)
            return f"{int(aid):04d}", f"{int(bid):04d}"
        m = PAT_SIMPLE.search(s_no_space)
        if m:
            aid = m.group(1)
            return f"{int(aid):04d}", None
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
                # 窗口里也要求包含 '+'（或 'D+'）来避免误报
                if ("+" not in joined) and not any("+" in tk for tk in window_tokens):
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
        # 防止覆盖：追加 _1, _2 ...
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


def process_one_image(img_path: Path):
    try:
        result_list = ocr.predict(str(img_path))
        all_texts, all_boxes = [], []
        for res in result_list:
            data = res.json
            inner = data.get("res", {})
            texts = inner.get("rec_texts", []) or []
            boxes = inner.get("rec_boxes", []) or inner.get("rec_polys", []) or []
            all_texts.extend(texts)
            all_boxes.extend(boxes)

        if not all_texts:
            # 无文本 → Other
            dst = OTHER_DIR / img_path.name
            safe_copy(img_path, dst)
            return False, None

        ids = find_ids_from_tokens(all_texts, all_boxes)
        if ids is None:
            # 未匹配到 → Other
            dst = OTHER_DIR / img_path.name
            safe_copy(img_path, dst)
            return False, None

        aid, bid = ids
        if bid:
            new_name = f"C{aid}-P{bid}.jpg"
        else:
            new_name = f"C{aid}.jpg"
        dst = CHAPTER_DIR / new_name
        safe_copy(img_path, dst)
        return True, (aid, bid)
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
    ok, fail = 0, 0
    for p in imgs:
        logging.info(f"当前还剩余:{len(imgs) - ok - fail},正在处理:{p}")
        hit, ids = process_one_image(p)
        if hit:
            ok += 1
        else:
            fail += 1
    logging.info(f"完成，识别成功 {ok} 张，未匹配 {fail} 张。输出目录：{DEST_ROOT}")


if __name__ == "__main__":
    main()
