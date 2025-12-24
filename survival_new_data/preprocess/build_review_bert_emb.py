from __future__ import annotations

"""
使用 BERT 模型（bert-base-multilingual-cased）对整合评论文本进行向量化。

特性与约定：
- 输入：`openrice/整合餐厅评论信息.csv`
- 粒度：逐条评论（每条评论 -> 一个向量），会过滤空文本（NaN / 纯空白）
- 输出：一个 Parquet 文件，默认路径为 `<project-root>/survival_hsk/data/review_bert_emb.parquet`
  - 字段包括：
    - review_id
    - restaurant_id
    - （可选）review_date（标准化为 YYYY-MM-DD 字符串）
    - bert_emb_0 ... bert_emb_{hidden_size-1}
- 本脚本不修改任何原始 CSV，只做离线特征抽取。

依赖：
- 需要安装 PyTorch 和 transformers：
  pip install torch transformers
- 默认优先使用 GPU（如可用），否则退回 CPU，建议在有 GPU 的环境中运行。
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


def get_project_root() -> Path:
    """
    返回项目根目录（包含 `openrice/` 的文件夹）。
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "openrice").is_dir():
            return parent
    # 兜底：假设当前文件位于 <root>/survival_new_data/preprocess/
    return current.parents[2]


def get_reviews_source_path(root: Optional[Path] = None) -> Path:
    """
    返回整合评论原始 CSV 的路径。

    优先从环境变量 OPENRICE_DIR 中读取原始 openrice 数据目录：
        OPENRICE_DIR=/data/openrice
    则评论文件为：
        /data/openrice/整合餐厅评论信息.csv
    若未设置，则默认使用 <project-root>/openrice/整合餐厅评论信息.csv。
    """
    openrice_dir_env = os.environ.get("OPENRICE_DIR")
    if openrice_dir_env:
        openrice_dir = Path(openrice_dir_env)
        return openrice_dir / "整合餐厅评论信息.csv"

    if root is None:
        root = get_project_root()
    return root / "openrice" / "整合餐厅评论信息.csv"


def get_output_path(
    root: Optional[Path] = None,
    out_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    返回评论 BERT 向量结果的输出路径（单个 Parquet 文件）。

    优先级：
    1. 若显式提供 out_path，则直接使用该完整文件路径；
    2. 若提供 output_dir，则在该目录下生成 `review_bert_emb.parquet`；
    3. 否则默认输出到 <project-root>/survival_hsk/data/review_bert_emb.parquet。
    """
    if out_path is not None and output_dir is not None:
        raise ValueError("out_path 和 output_dir 不能同时指定。")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    if root is None:
        root = get_project_root()

    if output_dir is not None:
        data_dir = Path(output_dir)
    else:
        data_dir = root / "survival_hsk" / "data"

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "review_bert_emb.parquet"


def load_bert_model(
    model_name: str = "bert-base-multilingual-cased",
) -> tuple[AutoTokenizer, AutoModel, torch.device]:
    """
    加载 BERT 分词器与模型，并自动选择设备（CUDA 优先）。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    # 打印当前使用的设备，便于在日志中确认是否使用了 GPU
    print(f"使用设备: {device}")
    return tokenizer, model, device


def iter_review_chunks(
    src: Path,
    usecols: List[str],
    chunk_size: int,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    按块迭代读取评论数据，支持 max_rows 调试限制。
    """
    rows_read = 0
    for chunk in pd.read_csv(src, usecols=usecols, chunksize=chunk_size):
        if max_rows is not None:
            remaining = max_rows - rows_read
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
        rows_read += len(chunk)
        yield chunk


def encode_batch(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int,
) -> np.ndarray:
    """
    对一批文本进行编码，返回 numpy 数组格式的向量。
    这里优先使用 BERT 的 pooler_output（[CLS] 向量）作为句向量表示；
    若模型未提供，则退回到最后一层 hidden_state 的平均池化。
    """
    if not texts:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output
        else:
            # 兜底：对最后一层 hidden_state 做加权平均池化
            last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]
            attention_mask = encoded["attention_mask"].unsqueeze(-1)  # [batch, seq_len, 1]
            masked = last_hidden * attention_mask
            length = attention_mask.sum(dim=1).clamp(min=1)
            emb = masked.sum(dim=1) / length

    return emb.cpu().numpy().astype(np.float32)


def build_bert_embeddings(
    max_rows: Optional[int],
    chunk_size: int,
    batch_size: int,
    max_length: int,
    total_rows_hint: Optional[int] = None,
    output_dir: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> None:
    """
    主过程：流式读取评论，过滤空文本，使用 BERT 生成向量，并以 Parquet 形式按批写盘。

    参数：
        max_rows: 仅用于调试，最多读取的原始评论行数；为 None 时处理全部。
        chunk_size: 分块读取 CSV 的块大小。
        batch_size: BERT 编码时的 batch 大小。
        max_length: BERT 编码时的最大 token 长度。
        total_rows_hint: （可选）总评论行数，用于打印进度百分比；为 None 时仅打印已处理数量。
        output_dir: （可选）向量结果输出目录；为 None 时使用默认 `<project-root>/survival_hsk/data`。
        out_path: （可选）完整输出文件路径；若设置则优先级最高。
    """
    root = get_project_root()
    src = get_reviews_source_path(root)
    output_path = get_output_path(root, out_path=out_path, output_dir=output_dir)

    if not src.exists():
        raise FileNotFoundError(f"找不到评论原始文件：{src}")

    print(f"评论原始文件: {src}")
    print(f"输出 Parquet: {output_path}")

    # 仅保留后续需要用于索引与分析的列
    usecols = ["review_id", "restaurant_id", "review_date", "review_text"]

    print("加载 BERT 模型（bert-base-multilingual-cased）...")
    tokenizer, model, device = load_bert_model()
    hidden_size = model.config.hidden_size
    print(f"模型隐藏维度: {hidden_size}")

    writer: Optional[pq.ParquetWriter] = None
    total_kept = 0
    total_rows = 0
    pbar: Optional[tqdm] = None

    try:
        # 若提供 total_rows_hint，则用于设定 tqdm 的总长度；否则为不定长进度条
        if total_rows_hint:
            pbar = tqdm(total=total_rows_hint, desc="编码评论", unit="行")
        else:
            pbar = tqdm(desc="编码评论", unit="行")

        for chunk in iter_review_chunks(src, usecols, chunk_size, max_rows):
            rows_this_chunk = len(chunk)
            total_rows += rows_this_chunk
            if pbar is not None:
                pbar.update(rows_this_chunk)

            # 过滤空文本：去除 NaN 和纯空白
            chunk["review_text"] = chunk["review_text"].astype(str)
            chunk["review_text_stripped"] = chunk["review_text"].str.strip()
            chunk = chunk[chunk["review_text_stripped"] != ""]

            if chunk.empty:
                continue

            # 可选：解析日期（统一为 YYYY-MM-DD 字符串）
            chunk["review_date"] = pd.to_datetime(
                chunk["review_date"], errors="coerce"
            )
            chunk["review_date"] = chunk["review_date"].dt.strftime("%Y-%m-%d")

            texts = chunk["review_text_stripped"].tolist()
            review_ids = chunk["review_id"].tolist()
            restaurant_ids = chunk["restaurant_id"].tolist()
            dates = chunk["review_date"].tolist()

            # 按 batch 编码
            start = 0
            while start < len(texts):
                end = min(start + batch_size, len(texts))
                batch_texts = texts[start:end]
                batch_review_ids = review_ids[start:end]
                batch_restaurant_ids = restaurant_ids[start:end]
                batch_dates = dates[start:end]

                emb = encode_batch(
                    batch_texts,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=max_length,
                )

                # 组装为 DataFrame
                data: Dict[str, Any] = {
                    "review_id": batch_review_ids,
                    "restaurant_id": batch_restaurant_ids,
                    "review_date": batch_dates,
                }
                for i in range(hidden_size):
                    data[f"bert_emb_{i}"] = emb[:, i]

                df_batch = pd.DataFrame(data)
                table = pa.Table.from_pandas(df_batch, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)

                total_kept += len(df_batch)
                start = end

            if total_rows_hint:
                percent = min(100.0, total_rows / total_rows_hint * 100.0)
                print(
                    f"已处理原始评论行数 {total_rows} / {total_rows_hint} "
                    f"({percent:.2f}%), "
                    f"文本非空并完成向量化的评论数: {total_kept}",
                    flush=True,
                )
            else:
                print(
                    f"已处理原始评论行数 {total_rows}, "
                    f"文本非空并完成向量化的评论数: {total_kept}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
        if pbar is not None:
            pbar.close()

    print("BERT 向量化完成。")
    print(f"最终写入的评论向量条数: {total_kept}")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="使用 bert-base-multilingual-cased 对整合评论文本进行向量化（离线脚本）。"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="仅用于调试：最多读取的原始评论行数，为 None 时处理全部。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="分块读取 CSV 的块大小。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="BERT 编码时的 batch 大小。",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="BERT 编码时的最大 token 长度（过长文本将被截断）。",
    )
    parser.add_argument(
        "--total-rows-hint",
        type=int,
        default=None,
        help="（可选）总评论行数，用于显示进度百分比。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="（可选）输出目录；若不指定则默认写入 <project-root>/survival_hsk/data。",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="（可选）完整输出文件路径，优先级高于 --output-dir。",
    )
    return parser.parse_args()


def main() -> None:
    """
    脚本入口。
    """
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None
    out_path = Path(args.out_path) if args.out_path else None

    build_bert_embeddings(
        max_rows=args.max_rows,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        max_length=args.max_length,
        total_rows_hint=args.total_rows_hint,
        output_dir=output_dir,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()


