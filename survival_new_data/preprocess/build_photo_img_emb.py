from __future__ import annotations

"""
将下载的 OpenRice 评论图片向量化，并输出 photo 级别的图片 embedding（每张图一行）。

输入（默认）：
- openrice_images/selected_reviews.csv（由 scripts/download_openrice_images.py 生成）
- openrice_images/<photo_id>.<ext>（下载的图片文件）

输出（默认）：
- <project-root>/survival_hsk/data/photo_img_emb.parquet
  字段：photo_id, review_id, img_emb_0..img_emb_{img_dim-1}

用法示例：
  python -m survival_new_data.preprocess.build_photo_img_emb ^
    --image-dir openrice_images ^
    --manifest openrice_images/selected_reviews.csv ^
    --max-images-per-review 1 ^
    --out-dir artifacts
"""

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
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


def _get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "openrice").is_dir():
            return parent
    return current.parents[2]


def _default_output_path() -> Path:
    return _get_project_root() / "survival_hsk" / "data" / "photo_img_emb.parquet"


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
class PhotoItem:
    photo_id: str
    review_id: str
    path: Path


def _iter_photos_from_manifest(
    manifest_path: Path,
    *,
    image_dir: Path,
    max_images_per_review: int,
) -> Iterable[PhotoItem]:
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
            for url, photo_id in plan:
                photo_id = str(photo_id).strip()
                if not photo_id:
                    continue
                p = _find_existing_image(image_dir, photo_id, url)
                if p is None:
                    continue
                yield PhotoItem(photo_id=photo_id, review_id=review_id, path=p)


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


def _encode_one_image(
    model: nn.Module,
    preprocess,
    path: Path,
    *,
    device: torch.device,
) -> Optional[np.ndarray]:
    try:
        img = Image.open(path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            y = model(x).detach().float().cpu().numpy()
        return y[0].astype(np.float32, copy=False)
    except Exception:
        return None


def _load_existing_photo_ids(out_path: Path) -> Set[str]:
    if not out_path.exists():
        return set()
    try:
        dataset = ds.dataset(str(out_path), format="parquet")
        table = dataset.to_table(columns=["photo_id"])
        df = table.to_pandas()
        if "photo_id" not in df.columns:
            return set()
        return set(df["photo_id"].astype(str).tolist())
    except Exception:
        return set()


def _append_row(
    *,
    writer: Optional[pq.ParquetWriter],
    out_path: Path,
    photo_id: str,
    review_id: str,
    emb: np.ndarray,
) -> pq.ParquetWriter:
    # pyarrow>=17: Table.from_pydict expects array-like values (scalars will error).
    data: Dict[str, object] = {"photo_id": [str(photo_id)], "review_id": [str(review_id)]}
    for i in range(int(emb.shape[0])):
        data[f"img_emb_{i}"] = [float(emb[i])]
    table = pa.Table.from_pydict(data)
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema)
    writer.write_table(table)
    return writer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vectorize downloaded OpenRice images to photo-level embeddings.")
    p.add_argument("--image-dir", type=str, default="openrice_images", help="图片目录（包含 photo_id 文件）")
    p.add_argument("--manifest", type=str, default="openrice_images/selected_reviews.csv", help="selected_reviews.csv 路径")
    p.add_argument("--out", type=str, default="", help="输出 parquet 文件路径（优先级高于 --out-dir）")
    p.add_argument("--out-dir", type=str, default="", help="输出目录（会写入 <out_dir>/photo_img_emb.parquet）")
    p.add_argument("--max-images-per-review", type=int, default=1, help="每条评论最多使用多少张图片（需与下载时一致）")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="auto 优先使用 CUDA")
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
        out_path = Path(args.out_dir) / "photo_img_emb.parquet"
    else:
        out_path = _default_output_path()
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

    existing = _load_existing_photo_ids(out_path)
    if existing:
        LOGGER.info("resume: loaded %d existing photo_id from %s", len(existing), out_path)

    writer: Optional[pq.ParquetWriter] = None
    kept = 0
    skipped = 0
    missing = 0
    errors = 0

    try:
        for item in _iter_photos_from_manifest(
            manifest,
            image_dir=image_dir,
            max_images_per_review=int(args.max_images_per_review),
        ):
            if item.photo_id in existing:
                skipped += 1
                continue

            LOGGER.info("encoding photo_id=%s review_id=%s path=%s", item.photo_id, item.review_id, item.path)
            emb = _encode_one_image(model, preprocess, item.path, device=device)
            if emb is None:
                errors += 1
                LOGGER.warning("failed photo_id=%s review_id=%s path=%s", item.photo_id, item.review_id, item.path)
                continue

            writer = _append_row(
                writer=writer,
                out_path=out_path,
                photo_id=item.photo_id,
                review_id=item.review_id,
                emb=emb,
            )
            existing.add(item.photo_id)
            kept += 1
            LOGGER.info("wrote photo_id=%s review_id=%s", item.photo_id, item.review_id)
            if kept % 5000 == 0:
                LOGGER.info("kept=%d skipped=%d missing=%d errors=%d", kept, skipped, missing, errors)

    finally:
        if writer is not None:
            writer.close()

    LOGGER.info("DONE out=%s kept=%d skipped=%d missing=%d errors=%d", out_path, kept, skipped, missing, errors)


if __name__ == "__main__":
    main()
