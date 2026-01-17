from __future__ import annotations

import argparse
import csv
import heapq
import logging
import math
import os
import random
import threading
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterable, List, Optional, Set, Tuple

import pandas as pd

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

KNOWN_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")

_thread_local = threading.local()


def _get_requests_session():
    if requests is None:
        return None
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def _as_int(value: object) -> int:
    if value is None:
        return 0
    try:
        if isinstance(value, float) and math.isnan(value):
            return 0
        return int(float(value))
    except Exception:
        return 0


def _as_str(value: object) -> str:
    if value is None:
        return ""
    s = str(value)
    if s.lower() == "nan":
        return ""
    return s


def create_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_img_entry(entry: str) -> str:
    entry = entry.strip()
    if "|" in entry:
        return entry.split("|", 1)[0].strip()
    return entry


def extract_image_plan(
    review_imgsrc: str,
    review_photo_id: str,
    max_images: int = 1,
    logger: Optional[logging.Logger] = None,
    review_id: str = "UNKNOWN",
) -> List[Tuple[str, str]]:
    raw_urls = [
        normalize_img_entry(part)
        for part in _as_str(review_imgsrc).split("\n")
        if part and _as_str(part).strip()
    ]
    raw_ids = [
        _as_str(part).strip()
        for part in _as_str(review_photo_id).split(";")
        if part and _as_str(part).strip()
    ]

    if logger is not None and len(raw_urls) != len(raw_ids):
        logger.warning(
            "图片 URL 与图片 ID 数量不一致，review_id=%s，urls=%d，ids=%d",
            review_id,
            len(raw_urls),
            len(raw_ids),
        )

    if max_images <= 0:
        limit = min(len(raw_urls), len(raw_ids))
    else:
        limit = min(max_images, len(raw_urls), len(raw_ids))
    return [(raw_urls[i], raw_ids[i]) for i in range(limit)]


def build_image_path(out_dir: Path, photo_id: str, img_url: str) -> Path:
    ext = os.path.splitext(img_url.split("?", 1)[0])[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
        ext = ".jpg"
    return out_dir / f"{photo_id}{ext}"


def scan_existing_photo_ids(out_dir: Path, logger: logging.Logger) -> Set[str]:
    """Return a set of photo_id derived from filenames in out_dir.

    Uses Path.stem so duplicates with different extensions are treated as the same photo.
    """
    photo_ids: Set[str] = set()
    if not out_dir.exists():
        return photo_ids
    try:
        for entry in out_dir.iterdir():
            if entry.is_file():
                stem = entry.stem.strip()
                if stem:
                    photo_ids.add(stem)
    except Exception as exc:
        logger.warning("Failed to scan existing images in %s: %s", out_dir, exc)
    return photo_ids


def photo_id_already_downloaded(out_dir: Path, photo_id: str, existing_photo_ids: Optional[Set[str]]) -> bool:
    if existing_photo_ids is not None and photo_id in existing_photo_ids:
        return True
    for ext in KNOWN_IMAGE_EXTS:
        if (out_dir / f"{photo_id}{ext}").exists():
            return True
    return False


def download_image(url: str, save_path: Path, max_retries: int, logger: logging.Logger) -> bool:
    for attempt in range(max_retries):
        try:
            if requests is not None:
                session = _get_requests_session()
                resp = session.get(url, timeout=20, stream=True, headers=HEADERS)  # type: ignore[union-attr]
                status = resp.status_code
                if status == 200:
                    with save_path.open("wb") as f:
                        for chunk in resp.iter_content(1024):
                            if chunk:
                                f.write(chunk)
                    logger.info("成功下载: %s", save_path)
                    return True
            else:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=20) as r:
                    status = getattr(r, "status", 200)
                    if status == 200:
                        with save_path.open("wb") as f:
                            f.write(r.read())
                        logger.info("成功下载: %s", save_path)
                        return True

            if status in (504, 503, 429):
                wait = (attempt + 1) * 2
                logger.warning("状态码 %s，等待 %.1fs 后重试，URL: %s", status, wait, url)
                time.sleep(wait)
            else:
                logger.warning("下载失败，状态码: %s，URL: %s", status, url)
                time.sleep(2)
        except Exception as exc:
            wait = (attempt + 1) * 1.5
            logger.warning(
                "下载尝试 %d 失败: %s，等待 %.1fs 后重试，URL: %s",
                attempt + 1,
                exc,
                wait,
                url,
            )
            time.sleep(wait)

    logger.error("超过最大重试次数，下载失败: %s", url)
    return False


@dataclass(frozen=True)
class SelectedReview:
    restaurant_id: str
    review_id: str
    score: float
    review_date_key: int
    review_photo_id: str
    review_imgsrc: str


def _date_key(value: object) -> int:
    s = _as_str(value).strip()
    if not s:
        return -1
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return -1
        return int(dt.value // 1_000_000_000)  # seconds
    except Exception:
        return -1


def compute_hotness_score(
    watch: object,
    helpful: object,
    comment: object,
    w_watch: float,
    w_helpful: float,
    w_comment: float,
) -> float:
    w = max(0, _as_int(watch))
    h = max(0, _as_int(helpful))
    c = max(0, _as_int(comment))
    return (w_watch * math.log1p(w)) + (w_helpful * h) + (w_comment * c)


def iter_selected_reviews(
    csv_path: Path,
    per_restaurant_k: int,
    *,
    chunk_size: int,
    max_rows: Optional[int],
    w_watch: float,
    w_helpful: float,
    w_comment: float,
    require_photo_num: bool,
    logger: logging.Logger,
) -> List[SelectedReview]:
    if per_restaurant_k <= 0:
        raise ValueError("--per-restaurant-k 必须 > 0")

    usecols = [
        "restaurant_id",
        "review_id",
        "review_date",
        "review_photo_num",
        "review_photo_id",
        "review_imgsrc",
        "review_watch_num",
        "review_helpful_vote",
        "review_liuyan_num",
    ]

    from collections import defaultdict

    heaps: DefaultDict[str, List[Tuple[float, int, str, str, str]]] = defaultdict(list)
    rows_seen = 0

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, encoding="utf-8"):
        if max_rows is not None and rows_seen >= max_rows:
            break

        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]

        rows_seen += len(chunk)

        for row in chunk.itertuples(index=False):
            restaurant_id = _as_str(getattr(row, "restaurant_id", "")).strip()
            review_id = _as_str(getattr(row, "review_id", "")).strip()
            if not restaurant_id or not review_id:
                continue

            review_imgsrc = _as_str(getattr(row, "review_imgsrc", ""))
            review_photo_id = _as_str(getattr(row, "review_photo_id", ""))
            if not review_imgsrc or not review_photo_id:
                continue

            if require_photo_num:
                if _as_int(getattr(row, "review_photo_num", 0)) <= 0:
                    continue

            score = compute_hotness_score(
                getattr(row, "review_watch_num", 0),
                getattr(row, "review_helpful_vote", 0),
                getattr(row, "review_liuyan_num", 0),
                w_watch=w_watch,
                w_helpful=w_helpful,
                w_comment=w_comment,
            )
            date_key = _date_key(getattr(row, "review_date", ""))

            heap = heaps[restaurant_id]
            item = (score, date_key, review_id, review_photo_id, review_imgsrc)
            if len(heap) < per_restaurant_k:
                heapq.heappush(heap, item)
            else:
                if item > heap[0]:
                    heapq.heapreplace(heap, item)

        if rows_seen and rows_seen % (chunk_size * 5) == 0:
            logger.info("已扫描 %d 行，当前覆盖店铺数=%d", rows_seen, len(heaps))

    selected: List[SelectedReview] = []
    for rid, heap in heaps.items():
        for score, date_key, review_id, photo_id, imgsrc in heap:
            selected.append(
                SelectedReview(
                    restaurant_id=rid,
                    review_id=review_id,
                    score=float(score),
                    review_date_key=int(date_key),
                    review_photo_id=photo_id,
                    review_imgsrc=imgsrc,
                )
            )

    selected.sort(key=lambda r: (r.restaurant_id, -r.score, -r.review_date_key, r.review_id))
    return selected


def download_for_selected(
    selected: Iterable[SelectedReview],
    out_dir: Path,
    *,
    max_images_per_review: int,
    min_delay_s: float,
    max_delay_s: float,
    max_retries: int,
    workers: int,
    max_pending: int,
    scan_existing: bool,
    log_every_s: float,
    logger: logging.Logger,
) -> None:
    selected_list = list(selected)
    total_reviews = len(selected_list)

    requested_images = 0
    success_images = 0
    failed_images = 0
    skipped_existing = 0
    skipped_duplicate = 0
    reviews_seen = 0

    create_directory(out_dir)

    existing_photo_ids: Optional[Set[str]] = None
    if scan_existing:
        logger.info("Scanning existing images in %s ...", out_dir.resolve())
        existing_photo_ids = scan_existing_photo_ids(out_dir, logger=logger)
        logger.info("Found %d existing photo_id", len(existing_photo_ids))

    seen_photo_ids: Set[str] = set(existing_photo_ids or ())

    def worker(img_url: str, save_path: Path) -> bool:
        if min_delay_s > 0 or max_delay_s > 0:
            time.sleep(random.uniform(min_delay_s, max_delay_s))
        return download_image(img_url, save_path, max_retries=max_retries, logger=logger)

    if workers <= 0:
        workers = 1
    if max_pending <= 0:
        max_pending = max(50, workers * 20)

    start_ts = time.monotonic()
    last_log_ts = start_ts

    def log_progress(*, force: bool = False) -> None:
        nonlocal last_log_ts
        now = time.monotonic()
        if not force and log_every_s > 0 and (now - last_log_ts) < log_every_s:
            return
        last_log_ts = now

        done_images = success_images + failed_images
        elapsed = max(1e-6, now - start_ts)
        img_rate = done_images / elapsed
        review_pct = (reviews_seen / total_reviews * 100.0) if total_reviews else 100.0
        logger.info(
            "Progress: reviews %d/%d (%.2f%%) | images done=%d (ok=%d fail=%d) | queued=%d | requested=%d | skipped_exist=%d skipped_dup=%d | %.2f img/s",
            reviews_seen,
            total_reviews,
            review_pct,
            done_images,
            success_images,
            failed_images,
            len(futures),
            requested_images,
            skipped_existing,
            skipped_duplicate,
            img_rate,
        )

    futures: Set[Future] = set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for r in selected_list:
            reviews_seen += 1
            plan = extract_image_plan(
                r.review_imgsrc,
                r.review_photo_id,
                max_images=max_images_per_review,
                logger=logger,
                review_id=r.review_id,
            )
            if not plan:
                continue

            for img_url, photo_id in plan:
                if photo_id in seen_photo_ids:
                    skipped_duplicate += 1
                    continue

                if photo_id_already_downloaded(out_dir, photo_id, existing_photo_ids):
                    seen_photo_ids.add(photo_id)
                    skipped_existing += 1
                    continue

                seen_photo_ids.add(photo_id)
                save_path = build_image_path(out_dir, photo_id, img_url)
                requested_images += 1

                while len(futures) >= max_pending:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for fut in done:
                        try:
                            if fut.result():
                                success_images += 1
                            else:
                                failed_images += 1
                        except Exception as exc:
                            failed_images += 1
                            logger.warning("Download task failed: %s", exc)
                    log_progress()

                futures.add(executor.submit(worker, img_url, save_path))

            log_progress()

        if futures:
            done, _ = wait(futures)
            for fut in done:
                try:
                    if fut.result():
                        success_images += 1
                    else:
                        failed_images += 1
                except Exception as exc:
                    failed_images += 1
                    logger.warning("Download task failed: %s", exc)

    logger.info("===== 完成 =====")
    log_progress(force=True)
    logger.info("评论数: %d", reviews_seen)
    logger.info("请求图片数: %d", requested_images)
    logger.info("成功下载: %d", success_images)
    logger.info("失败: %d", failed_images)
    logger.info("跳过已存在: %d", skipped_existing)
    logger.info("跳过重复photo_id: %d", skipped_duplicate)
    if requested_images:
        logger.info("成功率: %.2f%%", success_images / requested_images * 100)


def write_selected_manifest(selected: List[SelectedReview], out_dir: Path) -> Path:
    create_directory(out_dir)
    path = out_dir / "selected_reviews.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "restaurant_id",
                "review_id",
                "score",
                "review_date_key",
                "review_photo_id",
                "review_imgsrc",
            ]
        )
        for r in selected:
            w.writerow(
                [
                    r.restaurant_id,
                    r.review_id,
                    f"{r.score:.6f}",
                    r.review_date_key,
                    r.review_photo_id,
                    r.review_imgsrc,
                ]
            )
    return path


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("openrice_image_downloader")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "从 openrice/整合餐厅评论信息.csv 里按“热度”筛选每个店铺 Top-K 评论，并下载其图片。"
        )
    )
    p.add_argument(
        "--csv",
        default="openrice/整合餐厅评论信息.csv",
        help="整合评论 CSV 路径",
    )
    p.add_argument(
        "--out-dir",
        default="openrice_images",
        help="图片输出目录（会自动创建）",
    )
    p.add_argument(
        "--per-restaurant-k",
        type=int,
        default=10,
        help="每个店铺最多选多少条“热度最高”的评论",
    )
    p.add_argument(
        "--max-images-per-review",
        type=int,
        default=1,
        help="每条评论最多下载多少张图（按 CSV 中顺序，<=0 表示下载该评论的全部图片）",
    )
    p.add_argument("--chunk-size", type=int, default=100_000, help="分块读取行数")
    p.add_argument("--max-rows", type=int, default=None, help="仅用于调试：最多扫描多少行")

    p.add_argument("--w-watch", type=float, default=1.0, help="观看数权重（使用 log1p）")
    p.add_argument("--w-helpful", type=float, default=3.0, help="有用票权重")
    p.add_argument("--w-comment", type=float, default=2.0, help="留言数权重")
    p.add_argument(
        "--no-require-photo-num",
        action="store_true",
        help="默认只处理 review_photo_num>0 的行；加这个参数会跳过该过滤",
    )

    p.add_argument("--min-delay-s", type=float, default=0.05, help="每次请求前最小随机延迟（秒）")
    p.add_argument("--max-delay-s", type=float, default=0.3, help="每次请求前最大随机延迟（秒）")
    p.add_argument("--max-retries", type=int, default=5, help="单张图片最大重试次数")
    p.add_argument("--workers", type=int, default=8, help="并发下载线程数（<=0 视为 1）")
    p.add_argument(
        "--max-pending",
        type=int,
        default=0,
        help="最多积压任务数（0 自适应），用于防止内存堆积",
    )
    p.add_argument(
        "--log-every-s",
        type=float,
        default=10.0,
        help="每隔多少秒打印一次进度（<=0 表示尽量频繁打印）",
    )
    p.add_argument(
        "--no-scan-existing",
        action="store_true",
        help="不预扫描 out-dir 中已存在的 photo_id（默认会扫描，用于更好跳过重复/已下载图片）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    create_directory(out_dir)
    logger = setup_logger(out_dir / "download.log")

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV：{csv_path}")

    logger.info("CSV: %s", csv_path)
    logger.info("图片输出目录: %s", out_dir.resolve())
    logger.info(
        "筛选：每店 Top-%d；每评论最多 %d 张图；chunk_size=%d",
        args.per_restaurant_k,
        args.max_images_per_review,
        args.chunk_size,
    )

    selected = iter_selected_reviews(
        csv_path,
        per_restaurant_k=args.per_restaurant_k,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
        w_watch=args.w_watch,
        w_helpful=args.w_helpful,
        w_comment=args.w_comment,
        require_photo_num=not args.no_require_photo_num,
        logger=logger,
    )
    manifest_path = write_selected_manifest(selected, out_dir=out_dir)
    logger.info("筛选完成：共选中 %d 条评论，清单写入 %s", len(selected), manifest_path)

    download_for_selected(
        selected,
        out_dir=out_dir,
        max_images_per_review=args.max_images_per_review,
        min_delay_s=min(args.min_delay_s, args.max_delay_s),
        max_delay_s=max(args.min_delay_s, args.max_delay_s),
        max_retries=args.max_retries,
        workers=args.workers,
        max_pending=args.max_pending,
        scan_existing=not args.no_scan_existing,
        log_every_s=args.log_every_s,
        logger=logger,
    )


if __name__ == "__main__":
    main()
