"""
GoEmotions embedding builder for a Modell-style representation-manifold experiment.

What this fixes compared with the original quick script:
- No API key in source code. Set OPENAI_API_KEY in your environment.
- Keeps metadata aligned with embeddings: original row, text, label id, label name.
- Uses stratified sampling instead of silently taking the first N examples.
- Can optionally embed label words/prompts for the label-word control experiment.

Windows PowerShell example:
    $env:OPENAI_API_KEY="sk-..."
    python goemotions_embed_pipeline.py `
        --data-dir "D:/pyprojects/m4r/data/goemotions1" `
        --max-total 5000 `
        --max-per-label 250 `
        --embed-label-words
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from openai import OpenAI


DEFAULT_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", 
]


def chunker(seq: List[str], size: int) -> Iterable[List[str]]:
    for pos in range(0, len(seq), size):
        yield seq[pos:pos + size]


def read_emotions(path: Path | None) -> list[str]:
    if path is not None and path.exists():
        labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if labels:
            return labels
    return DEFAULT_LABELS


def parse_single_label(value: object) -> int | None:
    """Return the single GoEmotions label id if exactly one id is present, else None."""
    nums = re.findall(r"\d+", str(value))
    if len(nums) != 1:
        return None
    return int(nums[0])


def load_single_label_goemotions(tsv_path: Path, emotions: list[str]) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", header=None, usecols=[0, 1], names=["text", "label_raw"])
    df["label_idx"] = df["label_raw"].apply(parse_single_label)
    df = df.dropna(subset=["label_idx"]).copy()
    df["label_idx"] = df["label_idx"].astype(int)
    df = df[df["label_idx"].between(0, len(emotions) - 1)].copy()
    df["label_name"] = df["label_idx"].map(lambda i: emotions[int(i)])
    df["original_row"] = df.index
    return df[["original_row", "text", "label_idx", "label_name"]].reset_index(drop=True)


def stratified_sample(
    df: pd.DataFrame,
    max_total: int | None,
    max_per_label: int | None,
    seed: int,
) -> pd.DataFrame:
    out = df.copy()
    if max_per_label is not None:
        out = (
            out.groupby("label_name", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), max_per_label), random_state=seed))
            .reset_index(drop=True)
        )
    if max_total is not None and len(out) > max_total:
        # Stratified-ish: sample within labels with weights proportional to available counts.
        out = (
            out.groupby("label_name", group_keys=False)
            .apply(lambda g: g.sample(frac=min(1.0, max_total / len(out)), random_state=seed))
            .reset_index(drop=True)
        )
        if len(out) > max_total:
            out = out.sample(n=max_total, random_state=seed).reset_index(drop=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def clean_texts(texts: Iterable[str], max_chars: int) -> list[str]:
    cleaned: list[str] = []
    for text in texts:
        s = str(text).replace("\n", " ").strip()
        cleaned.append(s[:max_chars])
    return cleaned


def get_embeddings(
    client: OpenAI,
    texts: list[str],
    model: str,
    batch_size: int,
    max_chars: int,
    sleep_seconds: float = 0.0,
) -> np.ndarray:
    texts = clean_texts(texts, max_chars=max_chars)
    rows: list[list[float]] = []
    total = len(texts)
    for start, chunk in enumerate(chunker(texts, batch_size)):
        batch_no = start + 1
        print(f"Embedding batch {batch_no}: {len(chunk)} texts")
        response = client.embeddings.create(input=chunk, model=model)
        rows.extend([item.embedding for item in response.data])
        if sleep_seconds:
            time.sleep(sleep_seconds)
    arr = np.asarray(rows, dtype=np.float32)
    if arr.shape[0] != total:
        raise RuntimeError(f"Expected {total} embeddings, got {arr.shape[0]}")
    return arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing train.tsv and optionally emotions.txt")
    parser.add_argument("--tsv", type=str, default="train.tsv")
    parser.add_argument("--emotions", type=str, default="emotions.txt")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model", type=str, default="text-embedding-3-large")
    parser.add_argument("--max-total", type=int, default=5000)
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-chars", type=int, default=8190)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--save-csv", action="store_true", help="Also save embeddings as a headerless CSV for compatibility")
    parser.add_argument("--text-template", type=str, default="{text}", help="Example: 'The emotion expressed is: {text}'")
    parser.add_argument("--embed-label-words", action="store_true", help="Also embed GoEmotions label names as a label-word baseline")
    parser.add_argument("--label-word-template", type=str, default="the emotion {label}")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Do not put keys in source code; set it in your shell instead.")
    client = OpenAI(api_key=api_key)

    data_dir = args.data_dir
    out_dir = args.out_dir or data_dir / "manifold_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    emotions = read_emotions(data_dir / args.emotions)
    single = load_single_label_goemotions(data_dir / args.tsv, emotions)
    sampled = stratified_sample(single, max_total=args.max_total, max_per_label=args.max_per_label, seed=args.seed)
    sampled["embedding_text"] = sampled["text"].map(lambda t: args.text_template.format(text=t))

    print("Single-label rows available:", len(single))
    print("Rows selected:", len(sampled))
    print("Selected label counts:")
    print(sampled["label_name"].value_counts().sort_index())

    metadata_path = out_dir / "metadata.csv"
    emb_path = out_dir / "embeddings.npy"
    sampled.to_csv(metadata_path, index=False)
    embeddings = get_embeddings(
        client=client,
        texts=sampled["embedding_text"].tolist(),
        model=args.model,
        batch_size=args.batch_size,
        max_chars=args.max_chars,
        sleep_seconds=args.sleep_seconds,
    )
    np.save(emb_path, embeddings)
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved embeddings: {emb_path} {embeddings.shape}")

    if args.save_csv:
        csv_path = out_dir / "embeddings.csv"
        pd.DataFrame(embeddings).to_csv(csv_path, index=False, header=False)
        print(f"Saved CSV embeddings: {csv_path}")

    if args.embed_label_words:
        label_rows = pd.DataFrame({
            "label_idx": list(range(len(emotions))),
            "label_name": emotions,
        })
        label_rows["embedding_text"] = label_rows["label_name"].map(lambda lab: args.label_word_template.format(label=lab))
        label_meta_path = out_dir / "label_word_metadata.csv"
        label_emb_path = out_dir / "label_word_embeddings.npy"
        label_rows.to_csv(label_meta_path, index=False)
        label_embeddings = get_embeddings(
            client=client,
            texts=label_rows["embedding_text"].tolist(),
            model=args.model,
            batch_size=args.batch_size,
            max_chars=args.max_chars,
            sleep_seconds=args.sleep_seconds,
        )
        np.save(label_emb_path, label_embeddings)
        print(f"Saved label-word metadata: {label_meta_path}")
        print(f"Saved label-word embeddings: {label_emb_path} {label_embeddings.shape}")


if __name__ == "__main__":
    main()
