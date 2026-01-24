from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

_DEFAULT_STOPWORDS_EN = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "so",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "too",
    "up",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
    "yours",
}


def _clean_text(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = text.replace("\u00a0", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text


def _contains_cjk(token: str) -> bool:
    return _CJK_RE.search(token) is not None


def _read_stopwords(path: Path | None, *, include_default: bool) -> set[str]:
    words: set[str] = set(_DEFAULT_STOPWORDS_EN) if include_default else set()
    if path is None:
        return words
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w or w.startswith("#"):
                continue
            words.add(w)
    return words


class Tokenizer:
    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError


class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [m.group(0) for m in _TOKEN_RE.finditer(text)]


class CharNgramTokenizer(Tokenizer):
    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for tok in _TOKEN_RE.findall(text):
            if not tok:
                continue
            if _contains_cjk(tok) and len(tok) >= self.n:
                out.extend(tok[i : i + self.n] for i in range(0, len(tok) - self.n + 1))
            else:
                out.append(tok)
        return out


class JiebaTokenizer(Tokenizer):
    def __init__(self) -> None:
        import jieba  # type: ignore

        self._jieba = jieba

    def tokenize(self, text: str) -> list[str]:
        return [t.strip() for t in self._jieba.lcut(text, HMM=True) if t and t.strip()]


class PyCantoneseTokenizer(Tokenizer):
    def __init__(self) -> None:
        import pycantonese as pc  # type: ignore

        self._pc = pc

    def tokenize(self, text: str) -> list[str]:
        return [str(t).strip() for t in self._pc.segment(text) if str(t).strip()]


def build_tokenizer(mode: str, char_ngram: int) -> Tokenizer:
    mode = mode.lower().strip()
    if mode in {"pycantonese", "cantonese"}:
        return PyCantoneseTokenizer()
    if mode == "regex":
        return RegexTokenizer()
    if mode == "char":
        return CharNgramTokenizer(n=char_ngram)
    if mode == "jieba":
        return JiebaTokenizer()
    if mode == "auto":
        try:
            return PyCantoneseTokenizer()
        except Exception:
            try:
                return JiebaTokenizer()
            except Exception:
                return CharNgramTokenizer(n=char_ngram)
    raise ValueError(f"Unknown tokenizer mode: {mode}")


def iter_texts_from_csv(
    csv_path: Path,
    *,
    text_cols: list[str],
    chunksize: int,
    encoding: str,
) -> Iterable[str]:
    header = pd.read_csv(csv_path, nrows=0, encoding=encoding, engine="python")
    available = set(header.columns.tolist())
    selected = [c for c in text_cols if c in available]
    if not selected:
        raise ValueError(
            f"No requested text columns found in {csv_path.name}. "
            f"Requested={text_cols}, available={sorted(available)[:20]}..."
        )

    reader = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        encoding=encoding,
        usecols=selected,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        engine="python",
        on_bad_lines="skip",
    )
    for chunk in reader:
        for col in selected:
            for val in chunk[col]:
                if val:
                    yield val


@dataclass(frozen=True)
class FilterCfg:
    lowercase: bool
    min_token_len: int
    min_cjk_token_len: int


def tokenize_and_filter(
    text: str,
    *,
    tokenizer: Tokenizer,
    stopwords: set[str],
    cfg: FilterCfg,
) -> list[str]:
    text = _clean_text(text)
    if cfg.lowercase:
        text = text.lower()
    toks = tokenizer.tokenize(text)
    out: list[str] = []
    for tok in toks:
        tok = tok.strip()
        if not tok:
            continue
        stop_key = tok if _contains_cjk(tok) else tok.lower()
        if stop_key in stopwords:
            continue
        if _contains_cjk(tok):
            if len(tok) < cfg.min_cjk_token_len:
                continue
        else:
            if len(tok) < cfg.min_token_len:
                continue
        out.append(tok)
    return out


def sample_documents(
    texts: Iterable[str],
    *,
    max_docs: int,
    seed: int,
    tokenizer: Tokenizer,
    stopwords: set[str],
    filt: FilterCfg,
) -> list[str]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    for text in tqdm(texts, desc="Sampling docs", unit="doc"):
        tokens = tokenize_and_filter(text, tokenizer=tokenizer, stopwords=stopwords, cfg=filt)
        if not tokens:
            continue
        doc = " ".join(tokens)
        seen += 1
        if max_docs <= 0:
            reservoir.append(doc)
            continue
        if len(reservoir) < max_docs:
            reservoir.append(doc)
            continue
        j = rng.randrange(seen)
        if j < max_docs:
            reservoir[j] = doc
    return reservoir


def write_topics_csv(topics: dict[int, list[tuple[str, float]]], out_path: Path, topic_sizes: dict[int, int]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic_id", "topic_size", "rank", "token", "weight"])
        for topic_id in sorted(topics.keys()):
            if topic_id == -1:
                continue
            words = topics[topic_id] or []
            for rank, (token, weight) in enumerate(words, start=1):
                w.writerow([topic_id, topic_sizes.get(topic_id, 0), rank, token, weight])


def write_topic_info_csv(info_rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not info_rows:
        return
    fields = list(info_rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in info_rows:
            w.writerow(r)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Run BERTopic (or BERTopic + KMeans) per file on sampled review docs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", type=Path, default=Path("final_review_data"))
    p.add_argument("--glob", default="review_*.csv")
    p.add_argument("--text-cols", nargs="+", default=["review_text"])
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("--chunksize", type=int, default=50_000)
    p.add_argument(
        "--tokenizer",
        choices=["auto", "pycantonese", "cantonese", "jieba", "char", "regex"],
        default="auto",
    )
    p.add_argument("--char-ngram", type=int, default=2)
    p.add_argument("--stopwords", type=Path, default=None)
    p.add_argument("--no-default-stopwords", action="store_true")
    p.add_argument("--lowercase", action="store_true")
    p.add_argument("--min-token-len", type=int, default=2)
    p.add_argument("--min-cjk-token-len", type=int, default=2)
    p.add_argument("--max-docs", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-topics", type=int, default=0, help="If >0, use KMeans with fixed topic count.")
    p.add_argument("--embedding-model", default="paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/topics/bertopic"))
    args = p.parse_args(argv)

    try:
        from bertopic import BERTopic  # type: ignore
    except Exception as e:
        print(
            "BERTopic not installed. Try: pip install bertopic sentence-transformers umap-learn",
            file=sys.stderr,
        )
        raise

    from sentence_transformers import SentenceTransformer  # type: ignore

    files = sorted(Path(args.input_dir).glob(args.glob))
    if not files:
        print(f"No files matched: {Path(args.input_dir) / args.glob}", file=sys.stderr)
        return 2

    tokenizer = build_tokenizer(str(args.tokenizer), char_ngram=int(args.char_ngram))
    stopwords = _read_stopwords(args.stopwords, include_default=not bool(args.no_default_stopwords))
    filt = FilterCfg(
        lowercase=bool(args.lowercase),
        min_token_len=int(args.min_token_len),
        min_cjk_token_len=int(args.min_cjk_token_len),
    )

    embedder = SentenceTransformer(str(args.embedding_model))

    # Use BERTopic with either HDBSCAN (default) or a fixed-number clustering fallback (KMeans).
    hdbscan_model = None
    if int(args.n_topics) > 0:
        from sklearn.cluster import KMeans

        hdbscan_model = KMeans(n_clusters=int(args.n_topics), random_state=int(args.seed), n_init=10)

    for fp in files:
        print(f"\n==> {fp.name}")
        docs = sample_documents(
            iter_texts_from_csv(fp, text_cols=list(args.text_cols), chunksize=int(args.chunksize), encoding=str(args.encoding)),
            max_docs=int(args.max_docs),
            seed=int(args.seed),
            tokenizer=tokenizer,
            stopwords=stopwords,
            filt=filt,
        )
        if not docs:
            print("No documents after filtering; skipping.", file=sys.stderr)
            continue

        embeddings = embedder.encode(docs, batch_size=int(args.batch_size), show_progress_bar=True)

        topic_model = BERTopic(
            language="multilingual",
            calculate_probabilities=False,
            verbose=False,
            nr_topics=None,
            hdbscan_model=hdbscan_model,
        )
        topics, _ = topic_model.fit_transform(docs, embeddings)

        # Topic sizes
        size_counter: dict[int, int] = {}
        for t in topics:
            size_counter[int(t)] = size_counter.get(int(t), 0) + 1

        topics_dict = topic_model.get_topics()
        out_topics = Path(args.out_dir) / "by_file" / f"{fp.stem}_topics.csv"
        write_topics_csv(topics_dict, out_topics, topic_sizes=size_counter)
        print(f"Wrote: {out_topics}")

        info = topic_model.get_topic_info()
        out_info = Path(args.out_dir) / "by_file" / f"{fp.stem}_info.csv"
        write_topic_info_csv(info.to_dict(orient="records"), out_info)
        print(f"Wrote: {out_info}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
