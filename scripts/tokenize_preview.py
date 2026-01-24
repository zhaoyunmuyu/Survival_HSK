from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = text.replace("\u00a0", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text


def _contains_cjk(token: str) -> bool:
    return _CJK_RE.search(token) is not None


def _read_stopwords(path: Path | None) -> set[str]:
    if path is None:
        return set()
    words: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w or w.startswith("#"):
                continue
            words.add(w)
    return words


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


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Tokenize a raw review CSV and write a small preview CSV for inspection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True, help="Input CSV path")
    p.add_argument("--text-col", default="review_text", help="Column to tokenize")
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("--nrows", type=int, default=20, help="How many rows to preview")
    p.add_argument("--tokenizer", choices=["auto", "pycantonese", "cantonese", "jieba", "char", "regex"], default="auto")
    p.add_argument("--char-ngram", type=int, default=2)
    p.add_argument("--stopwords", type=Path, default=None, help="Optional stopwords file, one per line")
    p.add_argument("--apply-stopwords", action="store_true", help="If set, remove stopwords from output tokens")
    p.add_argument("--min-token-len", type=int, default=1, help="Min length for non-CJK tokens when filtering")
    p.add_argument("--min-cjk-token-len", type=int, default=1, help="Min length for CJK tokens when filtering")
    p.add_argument("--out", type=Path, default=None, help="Output CSV path (preview). Default under artifacts/tokenize_preview/")

    args = p.parse_args(argv)
    input_path: Path = args.input
    if not input_path.exists():
        print(f"Not found: {input_path}", file=sys.stderr)
        return 2

    out_path: Path
    if args.out is None:
        out_path = Path("artifacts/tokenize_preview") / f"{input_path.stem}_{args.tokenizer}.csv"
    else:
        out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(str(args.tokenizer), char_ngram=int(args.char_ngram))
    stopwords = _read_stopwords(args.stopwords)

    df = pd.read_csv(
        input_path,
        nrows=int(args.nrows),
        encoding=str(args.encoding),
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        engine="python",
        on_bad_lines="skip",
    )
    if args.text_col not in df.columns:
        print(f"Column not found: {args.text_col}. Available: {df.columns.tolist()}", file=sys.stderr)
        return 2

    # Keep a few helpful columns if present
    keep_cols = [c for c in ["review_id", "restaurant_id", "review_date", "review_year"] if c in df.columns]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([*keep_cols, "text", "tokens", "token_count"])
        for _, row in df.iterrows():
            text = _clean_text(str(row[args.text_col]))
            toks = [t.strip() for t in tokenizer.tokenize(text) if str(t).strip()]
            if args.apply_stopwords and stopwords:
                filtered: list[str] = []
                for t in toks:
                    key = t if _contains_cjk(t) else t.lower()
                    if key in stopwords:
                        continue
                    if _contains_cjk(t):
                        if len(t) < int(args.min_cjk_token_len):
                            continue
                    else:
                        if len(t) < int(args.min_token_len):
                            continue
                    filtered.append(t)
                toks = filtered
            w.writerow([*(row[c] for c in keep_cols), text, " ".join(toks), len(toks)])

    print(f"Wrote preview: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

