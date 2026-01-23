from __future__ import annotations

"""
Merge a sharded parquet dataset (directory of part-*.parquet) into a single parquet file.

Typical use case:
  - build_review_img_emb.py with --shard-size writes to review_img_emb.parts/
  - distill training prefers reading the parts directory, but sometimes a single file is desired.

Example:
  python -m survival_new_data.preprocess.merge_parquet_parts \
    --parts-dir /nvme02/gvm03-data/hsk/data/review_img_emb.parts \
    --out /nvme02/gvm03-data/hsk/data/review_img_emb.parquet \
    --compression zstd
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge parquet parts directory into a single parquet file.")
    p.add_argument("--parts-dir", type=str, required=True, help="Directory containing part-*.parquet files")
    p.add_argument("--out", type=str, required=True, help="Output parquet file path")
    p.add_argument("--compression", type=str, default="zstd", help="zstd/snappy/none (default zstd)")
    p.add_argument("--batch-size", type=int, default=65_536, help="Arrow scan batch size (rows)")
    p.add_argument("--row-group-size", type=int, default=65_536, help="Parquet row group size (rows)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists")
    p.add_argument("--skip-bad-fragments", action="store_true", help="Skip fragments that fail to read (prints warning)")
    return p.parse_args()


def _normalize_compression(value: str) -> Optional[str]:
    v = (value or "").strip().lower()
    if v in {"", "none", "null"}:
        return None
    return v


def merge_parts(*, parts_dir: Path, out_path: Path, compression: Optional[str], batch_size: int, row_group_size: int, overwrite: bool, skip_bad_fragments: bool) -> None:
    if not parts_dir.exists() or not parts_dir.is_dir():
        raise FileNotFoundError(f"parts-dir not found or not a directory: {parts_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if overwrite:
            out_path.unlink()
        else:
            raise FileExistsError(f"out already exists: {out_path} (use --overwrite)")

    dataset = ds.dataset(str(parts_dir), format="parquet")
    schema = dataset.schema

    scanner = dataset.scanner(batch_size=max(1, int(batch_size)), use_threads=True)
    writer: Optional[pq.ParquetWriter] = None
    rows_written = 0
    bad_fragments = 0

    try:
        for batch in scanner.to_batches():
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), schema, compression=compression)
            table = pa.Table.from_batches([batch], schema=schema)
            writer.write_table(table, row_group_size=max(1, int(row_group_size)))
            rows_written += table.num_rows
    except Exception as exc:
        # If scanner fails due to a single corrupt fragment, allow best-effort per-fragment retry.
        if not skip_bad_fragments:
            raise

        if writer is None:
            writer = pq.ParquetWriter(str(out_path), schema, compression=compression)

        for frag in dataset.get_fragments():
            try:
                frag_scanner = ds.Scanner.from_fragment(frag, schema=schema, batch_size=max(1, int(batch_size)), use_threads=True)
                for batch in frag_scanner.to_batches():
                    table = pa.Table.from_batches([batch], schema=schema)
                    writer.write_table(table, row_group_size=max(1, int(row_group_size)))
                    rows_written += table.num_rows
            except Exception as frag_exc:
                bad_fragments += 1
                print(f"[WARN] Skipping bad fragment: {getattr(frag, 'path', frag)}: {frag_exc}")

        if bad_fragments == 0:
            # Nothing was actually bad; re-raise original for visibility.
            raise exc
    finally:
        if writer is not None:
            writer.close()

    if rows_written <= 0:
        raise RuntimeError(f"Merge produced 0 rows (parts_dir={parts_dir})")

    # Footer validation
    _ = pq.read_metadata(str(out_path)).num_rows
    if bad_fragments:
        print(f"[DONE] out={out_path} rows={rows_written} bad_fragments={bad_fragments}")
    else:
        print(f"[DONE] out={out_path} rows={rows_written}")


def main() -> None:
    args = parse_args()
    parts_dir = Path(args.parts_dir)
    out_path = Path(args.out)
    merge_parts(
        parts_dir=parts_dir,
        out_path=out_path,
        compression=_normalize_compression(args.compression),
        batch_size=int(args.batch_size),
        row_group_size=int(args.row_group_size),
        overwrite=bool(args.overwrite),
        skip_bad_fragments=bool(args.skip_bad_fragments),
    )


if __name__ == "__main__":
    # Avoid buffering when run under nohup/tee.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()

