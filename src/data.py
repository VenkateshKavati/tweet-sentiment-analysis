"""Data loading and cleaning (mirrors Tasks 1-4 of the original notebook)."""
from pathlib import Path
import pandas as pd
from .preprocess import clean_text

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "twitter_training.csv"
COLUMNS = ["tweet_id", "entity", "sentiment", "text"]
LABELS = ["Negative", "Neutral", "Positive"]  # same order LabelEncoder produced


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=COLUMNS)


def load_clean(path=DATA_PATH, drop_irrelevant: bool = True) -> pd.DataFrame:
    df = load_raw(path)
    df = df.dropna(subset=["text"])            # <1% missing text
    df = df.drop_duplicates()                  # exact duplicate rows
    if drop_irrelevant:
        df = df[df["sentiment"] != "Irrelevant"]
    df = df.copy()
    df["length"] = df["text"].str.len()
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    df["label"] = df["sentiment"].map({l: i for i, l in enumerate(LABELS)})
    return df
