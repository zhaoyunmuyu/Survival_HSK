from __future__ import annotations

"""
边下载图片边向量化：持续扫描 image_dir，发现某条 review 的图片已下载齐，就立刻编码并追加写入 parquet。

适用场景：
- 你正在跑 scripts/download_openrice_images.py 在 openrice_images/ 下持续落盘图片；
- 同时启动本脚本，让 GPU/CPU 空闲时就把已下载的图片先向量化，缩短总耗时。

示例（另开一个终端并行运行）：
  python -m survival_new_data.preprocess.watch_review_img_emb ^
    --image-dir openrice_images ^
    --manifest openrice_images/selected_reviews.csv ^
    --out survival_hsk/data/review_img_emb.parquet ^
    --max-images-per-review 1 ^
    --device auto
"""

import argparse
import csv
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


LOGGER = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def _normalize_img_entry(entry: str) -> str:
    entry = (entry or "").strip()
    if "|" in entry:
        return entry.split("|", 1)[0].strip()
    return entry


def _extract_image_plan(review_imgsrc: str, review_photo_id: str, *, max_images: int) -> List[Tuple[str, str]]:
    raw_urls = [
        _normalize_img_entry(part)
        for part in (review_imgsrc or "").split("\n")
        if part and part.strip()
    ]
    raw_ids = [part.strip() for part in (review_photo_id or "").split(";") if part and part.strip()]
    if max_images <= 0:
        limit = min(len(raw_urls), len(raw_ids))
    else:
        limit = min(int(max_images), len(raw_urls), len(raw_ids))
    return [(raw_urls[i], raw_ids[i]) for i in range(limit)]


def _image_path_from_url(out_dir: Path, photo_id: str, img_url: str) -> Path:
    ext = os.path.splitext(str(img_url).split("?", 1)[0])[1].lower()
    if ext not in IMAGE_EXTS:
        ext = ".jpg"
    return out_dir / f"{photo_id}{ext}"


def _find_existing_image(out_dir: Path, photo_id: str, img_url: str) -> Optional[Path]:
    p = _image_path_from_url(out_dir, photo_id, img_url)
    if p.exists():
        return p
    for ext in IMAGE_EXTS:
        alt = out_dir / f"{photo_id}{ext}"
        if alt.exists():
            return alt
    return None


@dataclass(frozen=True)
class ReviewItem:
    review_id: str
    paths: Tuple[Path, ...]


def _iter_manifest_items(
    manifest_path: Path,
    *,
    image_dir: Path,
    max_images_per_review: int,
) -> Iterable[ReviewItem]:
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            review_id = str(row.get("review_id", "")).strip()
            if not review_id:
                continue
            plan = _extract_image_plan(
                str(row.get("review_imgsrc", "") or ""),
                str(row.get("review_photo_id", "") or ""),
                max_images=max_images_per_review,
            )
            paths: List[Path] = []
            for url, photo_id in plan:
                photo_id = str(photo_id).strip()
                if not photo_id:
                    continue
                p = _find_existing_image(image_dir, photo_id, url)
                if p is not None:
                    paths.append(p)
            yield ReviewItem(review_id=review_id, paths=tuple(paths))


def _load_resnet18_encoder(*, device: torch.device) -> Tuple[nn.Module, object, int]:
    try:  # torchvision>=0.13
        from torchvision.models import ResNet18_Weights  # type: ignore

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        preprocess = weights.transforms()
    except Exception:  # pragma: no cover
        model = resnet18(pretrained=True)
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    img_dim = 512
    return model, preprocess, img_dim


def _encode_images(
    model: nn.Module,
    preprocess,
    paths: Sequence[Path],
    *,
    device: torch.device,
    batch_size: int,
) -> Optional[np.ndarray]:
    if not paths:
        return None

    feats: List[np.ndarray] = []
    idx = 0
    while idx < len(paths):
        batch_paths = paths[idx : idx + int(batch_size)]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
            except Exception:
                continue
        if not images:
            idx += int(batch_size)
            continue
        x = torch.stack(images, dim=0).to(device)
        with torch.no_grad():
            y = model(x).detach().float().cpu().numpy()
        feats.append(y)
        idx += int(batch_size)

    if not feats:
        return None
    arr = np.concatenate(feats, axis=0)  # [N, D]
    return arr.mean(axis=0).astype(np.float32, copy=False)


def _load_existing_review_ids(out_path: Path) -> Set[str]:
    if not out_path.exists():
        return set()
    try:
        dataset = ds.dataset(str(out_path), format="parquet")
        table = dataset.to_table(columns=["review_id"])
        df = table.to_pandas()
        if "review_id" not in df.columns:
            return set()
        return set(df["review_id"].astype(str).tolist())
    except Exception:
        return set()


def _append_embedding_row(
    *,
    writer: Optional[pq.ParquetWriter],
    out_path: Path,
    review_id: str,
    emb: np.ndarray,
) -> pq.ParquetWriter:
    data: Dict[str, object] = {"review_id": str(review_id)}
    for i in range(int(emb.shape[0])):
        data[f"img_emb_{i}"] = float(emb[i])
    table = pa.Table.from_pydict(data)
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema)
    writer.write_table(table)
    return writer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch image_dir and incrementally write review image embeddings.")
    p.add_argument("--image-dir", type=str, default="openrice_images")
    p.add_argument("--manifest", type=str, default="openrice_images/selected_reviews.csv")
    p.add_argument("--out", type=str, default="", help="输出 parquet 文件路径（优先级高于 --out-dir）")
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="输出目录（会写入 <out_dir>/review_img_emb.parquet；若传了 --out 则忽略）",
    )
    p.add_argument("--max-images-per-review", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--poll-seconds", type=float, default=5.0, help="No new embeddings -> sleep this many seconds.")
    p.add_argument("--idle-exit-seconds", type=float, default=600.0, help="Exit if no progress for this long.")
    p.add_argument("--keep-missing", action="store_true", help="If no images found, write zero vector (not recommended while downloading).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)

    image_dir = Path(args.image_dir)
    manifest = Path(args.manifest)
    if args.out:
        out_path = Path(args.out)
    elif args.out_dir:
        out_path = Path(args.out_dir) / "review_img_emb.parquet"
    else:
        out_path = Path("survival_hsk") / "data" / "review_img_emb.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    LOGGER.info("device=%s", device)

    model, preprocess, img_dim = _load_resnet18_encoder(device=device)
    LOGGER.info("encoder=resnet18 img_dim=%d", img_dim)

    processed: Set[str] = _load_existing_review_ids(out_path)
    if processed:
        LOGGER.info("resume: loaded %d existing review_id from %s", len(processed), out_path)

    writer: Optional[pq.ParquetWriter] = None
    last_progress = time.time()
    total_written = 0
    missing = 0
    errors = 0

    try:
        while True:
            wrote_this_round = 0

            for item in _iter_manifest_items(
                manifest,
                image_dir=image_dir,
                max_images_per_review=int(args.max_images_per_review),
            ):
                if item.review_id in processed:
                    continue

                emb = None
                try:
                    if item.paths:
                        emb = _encode_images(
                            model,
                            preprocess,
                            item.paths,
                            device=device,
                            batch_size=int(args.batch_size),
                        )
                    if emb is None:
                        missing += 1
                        if not args.keep_missing:
                            continue
                        emb = np.zeros((img_dim,), dtype=np.float32)

                    writer = _append_embedding_row(writer=writer, out_path=out_path, review_id=item.review_id, emb=emb)
                    processed.add(item.review_id)
                    wrote_this_round += 1
                    total_written += 1
                    last_progress = time.time()
                    if total_written % 5000 == 0:
                        LOGGER.info("written=%d processed=%d missing_seen=%d errors=%d", total_written, len(processed), missing, errors)
                except Exception as exc:
                    errors += 1
                    if errors <= 10:
                        LOGGER.warning("failed review_id=%s: %s", item.review_id, exc)

            if wrote_this_round > 0:
                LOGGER.info("wrote %d new embeddings (total_written=%d)", wrote_this_round, total_written)
                continue

            idle = time.time() - last_progress
            if idle >= float(args.idle_exit_seconds):
                LOGGER.info("no progress for %.1fs; exiting. out=%s", idle, out_path)
                break

            time.sleep(max(0.2, float(args.poll_seconds)))

    finally:
        if writer is not None:
            writer.close()

    LOGGER.info("DONE out=%s total_written=%d missing_seen=%d errors=%d", out_path, total_written, missing, errors)


if __name__ == "__main__":
    main()
