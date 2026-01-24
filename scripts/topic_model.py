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
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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


def _stem_group(name: str) -> str:
    stem = name
    stem = re.sub(r"(\d+)(?=_[^_]+$)", "", stem)
    stem = re.sub(r"\d+$", "", stem)
    return stem


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


def fit_topics(
    docs: list[str],
    *,
    method: str,
    n_topics: int,
    max_features: int,
    random_state: int,
) -> tuple[list[str], list[list[tuple[str, float]]]]:
    if not docs:
        return [], []

    if method == "lda":
        vectorizer = CountVectorizer(
            max_features=max_features if max_features > 0 else None,
            token_pattern=r"(?u)\b\w+\b",
        )
        X = vectorizer.fit_transform(docs)
        model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            learning_method="online",
            batch_size=2048,
            n_jobs=-1,
        )
        model.fit(X)
        components = model.components_
    elif method == "nmf":
        vectorizer = TfidfVectorizer(
            max_features=max_features if max_features > 0 else None,
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True,
        )
        X = vectorizer.fit_transform(docs)
        model = NMF(
            n_components=n_topics,
            random_state=random_state,
            init="nndsvda",
            max_iter=400,
        )
        model.fit(X)
        components = model.components_
    else:
        raise ValueError(f"Unknown method: {method}")

    vocab = vectorizer.get_feature_names_out().tolist()

    topics: list[list[tuple[str, float]]] = []
    for k in range(components.shape[0]):
        row = components[k]
        idx = row.argsort()[::-1]
        top = [(vocab[i], float(row[i])) for i in idx[:30]]
        topics.append(top)
    return vocab, topics


def write_topics_csv(topics: list[list[tuple[str, float]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic_id", "rank", "token", "weight"])
        for topic_id, top in enumerate(topics):
            for rank, (token, weight) in enumerate(top, start=1):
                w.writerow([topic_id, rank, token, weight])


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Topic modeling (LDA/NMF) on review CSV files (sampled for scalability).",
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
    p.add_argument("--method", choices=["lda", "nmf"], default="nmf")
    p.add_argument("--n-topics", type=int, default=12)
    p.add_argument("--max-docs", type=int, default=20000, help="Reservoir-sample docs per file/group; <=0 means all")
    p.add_argument("--max-features", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/topics"))

    args = p.parse_args(argv)

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

    group_docs: dict[str, list[str]] = {}

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
        _, topics = fit_topics(
            docs,
            method=str(args.method),
            n_topics=int(args.n_topics),
            max_features=int(args.max_features),
            random_state=int(args.seed),
        )
        out_file = Path(args.out_dir) / str(args.method) / "by_file" / f"{fp.stem}_topics.csv"
        write_topics_csv(topics, out_file)
        print(f"Wrote: {out_file}")

        group = _stem_group(fp.stem)
        group_docs.setdefault(group, []).extend(docs)

    for group, docs in sorted(group_docs.items()):
        _, topics = fit_topics(
            docs,
            method=str(args.method),
            n_topics=int(args.n_topics),
            max_features=int(args.max_features),
            random_state=int(args.seed),
        )
        out_group = Path(args.out_dir) / str(args.method) / "by_group" / f"{group}_topics.csv"
        write_topics_csv(topics, out_group)
        print(f"Wrote: {out_group}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
