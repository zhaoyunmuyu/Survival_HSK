from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
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


def _clean_text(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = text.replace("\u00a0", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text


def _contains_cjk(token: str) -> bool:
    return _CJK_RE.search(token) is not None


class Tokenizer:
    def tokenize(self, text: str) -> Iterable[str]:
        raise NotImplementedError


class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str) -> Iterable[str]:
        for m in _TOKEN_RE.finditer(text):
            yield m.group(0)


class CharNgramTokenizer(Tokenizer):
    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n

    def tokenize(self, text: str) -> Iterable[str]:
        for tok in _TOKEN_RE.findall(text):
            if not tok:
                continue
            if _contains_cjk(tok) and len(tok) >= self.n:
                for i in range(0, len(tok) - self.n + 1):
                    yield tok[i : i + self.n]
            else:
                yield tok


class JiebaTokenizer(Tokenizer):
    def __init__(self) -> None:
        import jieba  # type: ignore

        self._jieba = jieba

    def tokenize(self, text: str) -> Iterable[str]:
        for tok in self._jieba.lcut(text, HMM=True):
            tok = tok.strip()
            if tok:
                yield tok


class PyCantoneseTokenizer(Tokenizer):
    def __init__(self) -> None:
        import pycantonese as pc  # type: ignore

        self._pc = pc

    def tokenize(self, text: str) -> Iterable[str]:
        for tok in self._pc.segment(text):
            tok = str(tok).strip()
            if tok:
                yield tok


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
            series = chunk[col]
            for val in series:
                if val:
                    yield val


@dataclass(frozen=True)
class TokenFilter:
    min_token_len: int
    min_cjk_token_len: int
    lowercase: bool


def iter_filtered_tokens(
    text: str,
    *,
    tokenizer: Tokenizer,
    stopwords: set[str],
    filt: TokenFilter,
) -> list[str]:
    text = _clean_text(text)
    if filt.lowercase:
        text = text.lower()

    out: list[str] = []
    for tok in tokenizer.tokenize(text):
        tok = tok.strip()
        if not tok:
            continue
        stop_key = tok if _contains_cjk(tok) else tok.lower()
        if stop_key in stopwords:
            continue
        if _contains_cjk(tok):
            if len(tok) < filt.min_cjk_token_len:
                continue
        else:
            if len(tok) < filt.min_token_len:
                continue
        out.append(tok)
    return out


def _stem_group(name: str) -> str:
    stem = name
    stem = re.sub(r"(\d+)(?=_[^_]+$)", "", stem)
    stem = re.sub(r"\d+$", "", stem)
    return stem


@dataclass
class TfidfStats:
    doc_count: int = 0
    total_tokens: int = 0
    tf: Counter[str] = None  # type: ignore[assignment]
    df: Counter[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tf is None:
            self.tf = Counter()
        if self.df is None:
            self.df = Counter()

    def update_doc(self, tokens: list[str]) -> None:
        self.doc_count += 1
        self.total_tokens += len(tokens)
        self.tf.update(tokens)
        self.df.update(set(tokens))

    def compute_rows(self, *, topn: int, min_tf: int, min_df: int) -> list[tuple[str, int, int, float, float, float]]:
        # Returns: (token, tf, df, tf_ratio, idf, tfidf)
        if self.doc_count <= 0 or self.total_tokens <= 0:
            return []

        rows: list[tuple[str, int, int, float, float, float]] = []
        n = self.doc_count
        total = self.total_tokens
        for tok, tf in self.tf.items():
            if tf < min_tf:
                continue
            df = int(self.df.get(tok, 0))
            if df < min_df:
                continue
            # sklearn-style smooth idf: log((n + 1) / (df + 1)) + 1
            idf = math.log((n + 1) / (df + 1)) + 1.0
            tf_ratio = tf / total
            tfidf = tf_ratio * idf
            rows.append((tok, int(tf), int(df), float(tf_ratio), float(idf), float(tfidf)))

        rows.sort(key=lambda x: x[-1], reverse=True)
        if topn > 0:
            rows = rows[:topn]
        return rows


def write_rows_csv(
    rows: list[tuple[str, int, int, float, float, float]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token", "tf", "df", "tf_ratio", "idf", "tfidf"])
        w.writerows(rows)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Compute TF-IDF-like token importance for review CSV files (aggregated by file/group).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", type=Path, default=Path("final_review_data"))
    p.add_argument("--glob", default="review_*.csv", help="Input file glob under input-dir")
    p.add_argument("--text-cols", nargs="+", default=["review_text"])
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("--chunksize", type=int, default=50_000)
    p.add_argument(
        "--tokenizer",
        choices=["auto", "pycantonese", "cantonese", "jieba", "char", "regex"],
        default="auto",
        help="Tokenization mode. auto=pycantonese if installed else jieba else char n-gram.",
    )
    p.add_argument("--char-ngram", type=int, default=2, help="Used when tokenizer=char/auto fallback")
    p.add_argument("--stopwords", type=Path, default=None, help="Optional stopwords file, one per line")
    p.add_argument(
        "--no-default-stopwords",
        action="store_true",
        help="Disable built-in English stopwords (e.g. the/and/of).",
    )
    p.add_argument("--lowercase", action="store_true")
    p.add_argument("--min-token-len", type=int, default=2, help="Min length for non-CJK tokens")
    p.add_argument("--min-cjk-token-len", type=int, default=2, help="Min length for CJK tokens")
    p.add_argument("--min-tf", type=int, default=20, help="Filter tokens with tf < min-tf")
    p.add_argument("--min-df", type=int, default=10, help="Filter tokens with df < min-df")
    p.add_argument("--topn", type=int, default=200, help="Top-N tokens to write; <=0 writes all")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/tfidf"))

    args = p.parse_args(argv)

    files = sorted(Path(args.input_dir).glob(args.glob))
    if not files:
        print(f"No files matched: {Path(args.input_dir) / args.glob}", file=sys.stderr)
        return 2

    tokenizer = build_tokenizer(args.tokenizer, char_ngram=int(args.char_ngram))
    stopwords = _read_stopwords(args.stopwords, include_default=not bool(args.no_default_stopwords))
    filt = TokenFilter(
        min_token_len=int(args.min_token_len),
        min_cjk_token_len=int(args.min_cjk_token_len),
        lowercase=bool(args.lowercase),
    )

    per_file: dict[str, TfidfStats] = {}
    per_group: dict[str, TfidfStats] = {}

    for fp in files:
        print(f"\n==> {fp.name}")
        stats = TfidfStats()
        group = _stem_group(fp.stem)
        group_stats = per_group.setdefault(group, TfidfStats())

        for text in tqdm(
            iter_texts_from_csv(fp, text_cols=list(args.text_cols), chunksize=int(args.chunksize), encoding=str(args.encoding)),
            desc="Docs",
            unit="doc",
        ):
            tokens = iter_filtered_tokens(text, tokenizer=tokenizer, stopwords=stopwords, filt=filt)
            if not tokens:
                continue
            stats.update_doc(tokens)
            group_stats.update_doc(tokens)

        per_file[fp.name] = stats

        rows = stats.compute_rows(topn=int(args.topn), min_tf=int(args.min_tf), min_df=int(args.min_df))
        out_file = Path(args.out_dir) / "by_file" / f"{fp.stem}_top.csv"
        write_rows_csv(rows, out_file)
        print(f"Wrote: {out_file}")

    # Group outputs + long table
    long_rows: list[tuple[str, str, int, int, float, float, float]] = []
    for group, stats in sorted(per_group.items()):
        rows = stats.compute_rows(topn=int(args.topn), min_tf=int(args.min_tf), min_df=int(args.min_df))
        out_group = Path(args.out_dir) / "by_group" / f"{group}_top.csv"
        write_rows_csv(rows, out_group)
        print(f"Wrote: {out_group}")
        for token, tf, df, tf_ratio, idf, tfidf in rows:
            long_rows.append((group, token, tf, df, tf_ratio, idf, tfidf))

    out_long = Path(args.out_dir) / "by_group" / "tfidf_by_group_long.csv"
    out_long.parent.mkdir(parents=True, exist_ok=True)
    with out_long.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "token", "tf", "df", "tf_ratio", "idf", "tfidf"])
        w.writerows(long_rows)
    print(f"\nWrote: {out_long}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
