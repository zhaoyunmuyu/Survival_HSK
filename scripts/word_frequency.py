from __future__ import annotations

import argparse
import csv
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
        # CJK: generate n-grams; Latin/digits: keep whole token.
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

    # Use Python engine to avoid rare C-engine buffer overflow on malformed rows.
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
            if col not in chunk.columns:
                continue
            series = chunk[col]
            for val in series:
                if not val:
                    continue
                yield val


@dataclass(frozen=True)
class CountConfig:
    min_token_len: int
    min_cjk_token_len: int
    min_count: int
    lowercase: bool


def count_tokens(
    texts: Iterable[str],
    tokenizer: Tokenizer,
    stopwords: set[str],
    config: CountConfig,
    total: int | None = None,
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in tqdm(texts, total=total, desc="Counting", unit="text"):
        text = _clean_text(text)
        if config.lowercase:
            text = text.lower()
        for tok in tokenizer.tokenize(text):
            tok = tok.strip()
            if not tok:
                continue
            stop_key = tok if _contains_cjk(tok) else tok.lower()
            if stop_key in stopwords:
                continue
            if _contains_cjk(tok):
                if len(tok) < config.min_cjk_token_len:
                    continue
            else:
                if len(tok) < config.min_token_len:
                    continue
            counter[tok] += 1
    if config.min_count > 1:
        counter = Counter({k: v for k, v in counter.items() if v >= config.min_count})
    return counter


def _stem_group(name: str) -> str:
    # review_all_decline1_dead.csv -> review_all_decline_dead
    # review_before_risk_survive2.csv -> review_before_risk_survive
    stem = name
    stem = re.sub(r"(\d+)(?=_[^_]+$)", "", stem)  # drop run number before last suffix
    stem = re.sub(r"\d+$", "", stem)  # drop trailing run number (e.g. survive1)
    return stem


def write_counter_csv(counter: Counter[str], out_path: Path, topn: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(counter.values()) or 1
    rows = counter.most_common(topn if topn > 0 else None)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token", "count", "ratio"])
        for token, count in rows:
            w.writerow([token, count, count / total])


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Compute token/word frequency for review CSV files.",
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
    p.add_argument("--min-count", type=int, default=2, help="Filter tokens with count < min-count")
    p.add_argument("--topn", type=int, default=200, help="Top-N tokens to write; <=0 writes all")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/word_freq"))

    args = p.parse_args(argv)
    in_dir: Path = args.input_dir
    files = sorted(in_dir.glob(args.glob))
    if not files:
        print(f"No files matched: {in_dir / args.glob}", file=sys.stderr)
        return 2

    tokenizer = build_tokenizer(args.tokenizer, char_ngram=args.char_ngram)
    stopwords = _read_stopwords(args.stopwords, include_default=not bool(args.no_default_stopwords))
    config = CountConfig(
        min_token_len=args.min_token_len,
        min_cjk_token_len=args.min_cjk_token_len,
        min_count=args.min_count,
        lowercase=bool(args.lowercase),
    )

    per_file: dict[str, Counter[str]] = {}
    per_group: dict[str, Counter[str]] = {}

    for fp in files:
        print(f"\n==> {fp.name}")
        texts = iter_texts_from_csv(
            fp,
            text_cols=list(args.text_cols),
            chunksize=int(args.chunksize),
            encoding=str(args.encoding),
        )
        counter = count_tokens(texts, tokenizer=tokenizer, stopwords=stopwords, config=config)
        per_file[fp.name] = counter

        group = _stem_group(fp.stem)
        per_group.setdefault(group, Counter()).update(counter)

        out_file = args.out_dir / "by_file" / f"{fp.stem}_top.csv"
        write_counter_csv(counter, out_file, topn=int(args.topn))
        print(f"Wrote: {out_file}")

    # Group summaries
    group_rows: list[tuple[str, str, int, float]] = []
    for group, counter in sorted(per_group.items()):
        total = sum(counter.values()) or 1
        for token, count in counter.most_common(int(args.topn) if int(args.topn) > 0 else None):
            group_rows.append((group, token, count, count / total))

        out_group = args.out_dir / "by_group" / f"{group}_top.csv"
        write_counter_csv(counter, out_group, topn=int(args.topn))
        print(f"Wrote: {out_group}")

    out_summary = args.out_dir / "by_group" / "word_freq_by_group_long.csv"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "token", "count", "ratio"])
        w.writerows(group_rows)
    print(f"\nWrote: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
