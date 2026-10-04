"""Train and compare models, then save the best pipeline + reports.

Usage:  python train.py             # fast linear models
        python train.py --with-rbf  # also run the original RBF-SVM baseline (slow)
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from src.data import LABELS, load_clean

ROOT = Path(__file__).parent
SEED = 156


def make_pipe(clf, ngram=(1, 2)):
    # Text is already cleaned in src.data, so the vectorizer only needs to tokenize.
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram, min_df=2, sublinear_tf=True)),
        ("clf", clf),
    ])


def main(with_rbf: bool):
    df = load_clean()
    print(f"Rows after cleaning: {len(df):,}")

    # Group split by tweet_id: paraphrases of one tweet never land in both train and test.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    tr, te = next(gss.split(df, df["label"], groups=df["tweet_id"]))
    X_tr, X_te = df.loc[tr, "clean_text"], df.loc[te, "clean_text"]
    y_tr, y_te = df.loc[tr, "label"], df.loc[te, "label"]

    candidates = {
        "Complement Naive Bayes": make_pipe(ComplementNB()),
        "Logistic Regression": make_pipe(LogisticRegression(max_iter=1000, C=4)),
        "Linear SVM (calibrated)": make_pipe(CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)),
    }
    if with_rbf:  # the original notebook's model, unigrams only, for comparison
        candidates["RBF SVM (original)"] = make_pipe(
            SVC(C=4, kernel="rbf", probability=True), ngram=(1, 1))

    results, fitted = {}, {}
    for name, pipe in candidates.items():
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        results[name] = {
            "accuracy": accuracy_score(y_te, pred),
            "macro_f1": f1_score(y_te, pred, average="macro"),
            "train_seconds": round(time.time() - t0, 1),
        }
        fitted[name] = pipe
        print(f"{name:28s} acc={results[name]['accuracy']:.3f} "
              f"f1={results[name]['macro_f1']:.3f} ({results[name]['train_seconds']}s)")

    # Light tuning on the best linear model (macro-F1, as in the original notebook)
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    if best_name == "Logistic Regression":
        gs = GridSearchCV(
            make_pipe(LogisticRegression(max_iter=1000)),
            {"clf__C": [1, 2, 4, 8, 16], "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)]},
            cv=3, scoring="f1_macro", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        tuned = gs.best_estimator_
        pred = tuned.predict(X_te)
        results["Logistic Regression (tuned)"] = {
            "accuracy": accuracy_score(y_te, pred),
            "macro_f1": f1_score(y_te, pred, average="macro"),
            "best_params": {k: str(v) for k, v in gs.best_params_.items()},
        }
        fitted["Logistic Regression (tuned)"] = tuned
        print("Tuned LR:", results["Logistic Regression (tuned)"])

    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best = fitted[best_name]
    pred = best.predict(X_te)
    print(f"\nBest model: {best_name}\n",
          classification_report(y_te, pred, target_names=LABELS))

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(best, ROOT / "models" / "sentiment_pipeline.joblib", compress=3)

    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=LABELS, cmap="Blues")
    plt.title(best_name)
    plt.savefig(ROOT / "reports" / "confusion_matrix.png", dpi=130, bbox_inches="tight")

    with open(ROOT / "reports" / "metrics.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "model_comparison": results,
            "per_class": classification_report(y_te, pred, target_names=LABELS, output_dict=True),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
        }, f, indent=2)
    print("Saved model and reports.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-rbf", action="store_true")
    main(ap.parse_args().with_rbf)
