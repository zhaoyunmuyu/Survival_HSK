from __future__ import annotations

"""
将下载的 OpenRice 评论图片向量化，并输出 review 级别的图片 embedding。

输入（默认）：
- openrice_images/selected_reviews.csv（由 scripts/download_openrice_images.py 生成）
- openrice_images/<photo_id>.<ext>（下载的图片文件）

输出（默认）：
- <project-root>/survival_hsk/data/review_img_emb.parquet
  字段：review_id, img_emb_0..img_emb_{img_dim-1}

用法示例（和下载脚本保持一致的 max-images-per-review）：
  python -m survival_new_data.preprocess.build_review_img_emb ^
    --image-dir openrice_images ^
    --manifest openrice_images/selected_reviews.csv ^
    --max-images-per-review 1

说明：
- 默认只写入“实际找到>=1张图片”的 review（缺失的 review 在训练/推理时会自动回退为全 0 向量）。
- 默认使用 torchvision 的 ResNet18 预训练模型，输出 512 维（与 distill 默认 img_dim=512 对齐）。
"""

import argparse
import csv
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


LOGGER = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")


def _get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "openrice").is_dir():
            return parent
    return current.parents[2]


def _default_output_path() -> Path:
    return _get_project_root() / "survival_hsk" / "data" / "review_img_emb.parquet"


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
    # fallback: 同一个 photo_id 可能被保存成不同后缀
    for ext in IMAGE_EXTS:
        alt = out_dir / f"{photo_id}{ext}"
        if alt.exists():
            return alt
    return None


@dataclass(frozen=True)
class ReviewImages:
    review_id: str
    paths: Tuple[Path, ...]


def _iter_reviews_from_manifest(
    manifest_path: Path,
    *,
    image_dir: Path,
    max_images_per_review: int,
    max_reviews: int = 0,
    max_total_images: int = 0,
) -> Iterable[ReviewImages]:
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        yielded = 0
        images_used = 0
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
            if max_total_images and images_used >= max_total_images:
                break
            if max_total_images:
                remaining = max_total_images - images_used
                if remaining <= 0:
                    break
                if len(paths) > remaining:
                    paths = paths[:remaining]
            yield ReviewImages(review_id=review_id, paths=tuple(paths))
            yielded += 1
            images_used += len(paths)
            if max_reviews and yielded >= max_reviews:
                break


def _load_resnet18_encoder(*, device: torch.device) -> Tuple[nn.Module, object, int]:
    # torchvision 版本差异：
    # - 新版：resnet18(weights=ResNet18_Weights.DEFAULT) + weights.transforms()
    # - 旧版：resnet18(pretrained=True) 但无 Weights enum
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


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vectorize downloaded OpenRice review images to review-level embeddings.")
    p.add_argument("--image-dir", type=str, default="openrice_images", help="图片目录（包含 photo_id 文件）")
    p.add_argument("--manifest", type=str, default="openrice_images/selected_reviews.csv", help="selected_reviews.csv 路径")
    p.add_argument("--out", type=str, default="", help="输出 parquet 文件路径（优先级高于 --out-dir）")
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="输出目录（会写入 <out_dir>/review_img_emb.parquet；若传了 --out 则忽略）",
    )
    p.add_argument("--max-images-per-review", type=int, default=1, help="每条评论最多使用多少张图片（需与下载时一致）")
    p.add_argument("--batch-size", type=int, default=32, help="图片编码 batch size（每条评论内部也会分批）")
    p.add_argument("--max-reviews", type=int, default=0, help="最多处理多少条评论（0 表示不限制，用于快速试跑）")
    p.add_argument("--max-total-images", type=int, default=0, help="最多向量化多少张图片（0 表示不限制，用于快速试跑）")
    p.add_argument("--write-batch-size", type=int, default=2048, help="Parquet 写入 batch 大小（越大越快，默认 2048）")
    p.add_argument("--compression", type=str, default="zstd", help="Parquet 压缩：zstd/snappy/none（默认 zstd）")
    p.add_argument(
        "--no-atomic-write",
        action="store_true",
        help="默认先写入临时 parquet 文件，成功后再替换为最终输出；加此参数将直接写 out",
    )
    p.add_argument("--log-each-review", action="store_true", help="打印每条评论的向量化过程（非常啰嗦）")
    p.add_argument("--log-image-paths", action="store_true", help="配合 --log-each-review：额外打印每张图片路径")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="auto 优先使用 CUDA")
    p.add_argument("--keep-missing", action="store_true", help="即使没有找到图片也写入全 0 向量（默认不写）")
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
        out_path = _default_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_path = out_path
    tmp_path: Optional[Path] = None
    if not getattr(args, "no_atomic_write", False):
        write_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path = write_path
        if write_path.exists():
            write_path.unlink()

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

    writer: Optional[pq.ParquetWriter] = None
    write_batch_size = max(1, int(getattr(args, "write_batch_size", 2048)))
    compression_raw = str(getattr(args, "compression", "zstd")).lower()
    compression = None if compression_raw in {"", "none", "null"} else compression_raw
    kept = 0
    missing = 0
    errors = 0

    pending_ids: List[str] = []
    pending_embs: List[np.ndarray] = []

    def _flush() -> None:
        nonlocal writer, kept, pending_ids, pending_embs
        if not pending_ids:
            return
        ids = pending_ids
        embs = np.stack(pending_embs, axis=0).astype(np.float32, copy=False)  # [N, D]
        pending_ids = []
        pending_embs = []

        data: Dict[str, object] = {"review_id": ids}
        for i in range(img_dim):
            data[f"img_emb_{i}"] = embs[:, i]
        table = pa.Table.from_pydict(data)
        if writer is None:
            writer = pq.ParquetWriter(write_path, table.schema, compression=compression)
        writer.write_table(table)
        kept += len(ids)

    processed_reviews = 0
    try:
        for item in _iter_reviews_from_manifest(
            manifest,
            image_dir=image_dir,
            max_images_per_review=int(args.max_images_per_review),
            max_reviews=int(getattr(args, "max_reviews", 0) or 0),
            max_total_images=int(getattr(args, "max_total_images", 0) or 0),
        ):
            processed_reviews += 1
            try:
                if getattr(args, "log_each_review", False):
                    LOGGER.info("review=%d review_id=%s images=%d", processed_reviews, item.review_id, len(item.paths))
                    if getattr(args, "log_image_paths", False):
                        for p in item.paths:
                            LOGGER.info("  img=%s", p)

                t0 = time.monotonic()
                emb = _encode_images(
                    model,
                    preprocess,
                    item.paths,
                    device=device,
                    batch_size=int(args.batch_size),
                )
                dt = time.monotonic() - t0
                if emb is None:
                    missing += 1
                    if getattr(args, "log_each_review", False):
                        LOGGER.info("review=%d review_id=%s missing_images dt=%.3fs", processed_reviews, item.review_id, dt)
                    if not args.keep_missing:
                        continue
                    emb = np.zeros((img_dim,), dtype=np.float32)

                pending_ids.append(str(item.review_id))
                pending_embs.append(np.asarray(emb, dtype=np.float32).reshape(-1))
                if len(pending_ids) >= write_batch_size:
                    _flush()

                if getattr(args, "log_each_review", False):
                    LOGGER.info("review=%d review_id=%s encoded=1 dt=%.3fs", processed_reviews, item.review_id, dt)

                written = kept + len(pending_ids)
                if written > 0 and written % 5000 == 0:
                    LOGGER.info("written=%d missing=%d errors=%d", written, missing, errors)
            except Exception as exc:
                errors += 1
                if errors <= 10:
                    LOGGER.warning("failed review_id=%s: %s", item.review_id, exc)

    finally:
        _flush()
        if writer is not None:
            writer.close()

    if tmp_path is not None and tmp_path.exists():
        with tmp_path.open("rb") as f:
            pf = pq.ParquetFile(f)
            _ = pf.metadata.num_rows
        os.replace(str(tmp_path), str(out_path))

    LOGGER.info("DONE out=%s written=%d missing=%d errors=%d", out_path, kept, missing, errors)


if __name__ == "__main__":
    main()
