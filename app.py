"""Streamlit dashboard: predict, batch-score, explore the data, inspect the model.

Run:  streamlit run app.py
"""
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

from src.data import LABELS, load_clean
from src.preprocess import clean_text

ROOT = Path(__file__).parent
COLORS = {"Negative": "#e5484d", "Neutral": "#8b8d98", "Positive": "#30a46c"}
EXAMPLES = [
    "I absolutely love the new Borderlands update, best patch ever!",
    "Servers are down again. Totally unacceptable, I want a refund.",
    "Just downloaded the game, will try it tonight.",
    "Not bad at all, but not great either.",
]

st.set_page_config(page_title="Tweet Sentiment Dashboard", page_icon="💬", layout="wide")


# ---------- cached loaders ----------
@st.cache_resource
def load_model():
    path = ROOT / "models" / "sentiment_pipeline.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_data():
    return load_clean()


@st.cache_data
def load_metrics():
    path = ROOT / "reports" / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def top_words(texts: tuple, n: int = 20) -> pd.DataFrame:
    counts = Counter(w for t in texts for w in t.split())
    return pd.DataFrame(counts.most_common(n), columns=["word", "count"])


def predict_proba(model, texts):
    return model.predict_proba([clean_text(t) for t in texts])


def word_influence(model, text, pred_idx):
    """Leave-one-word-out: how much does the predicted-class probability drop without each word?"""
    words = clean_text(text).split()
    if len(words) < 2:
        return pd.DataFrame(columns=["word", "influence"])
    base = model.predict_proba([" ".join(words)])[0][pred_idx]
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    probs = model.predict_proba(variants)[:, pred_idx]
    return pd.DataFrame({"word": words, "influence": base - probs}).sort_values("influence")


model, data, metrics = load_model(), load_data(), load_metrics()

with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider(
        "Confidence needed to commit to a label", 0.34, 0.90, 0.50, 0.01,
        help="Below this, the dashboard says 'Uncertain' instead of forcing a label. "
             "On held-out tweets, predictions under 0.50 confidence were right ~46% of the time, "
             "versus ~74% at 0.50 or above.")

st.title("💬 Tweet Sentiment Dashboard")
st.caption("Entity-level Twitter sentiment: Negative / Neutral / Positive. "
           "TF-IDF + linear model, trained on the Twitter entity sentiment dataset.")

if model is None:
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

tab_pred, tab_batch, tab_explore, tab_model = st.tabs(
    ["🔮 Predict", "📂 Batch scoring", "📊 Explore data", "🧪 Model report"])

# ---------- Predict ----------
with tab_pred:
    st.subheader("Try a tweet")
    ex = st.selectbox("Load an example", ["(write your own)"] + EXAMPLES)
    # A form sends the typed text and the click together. A bare st.button next to a text_area
    # can need two clicks, because the first click only "commits" the text box.
    with st.form("predict_form", clear_on_submit=False):
        text = st.text_area("Tweet text (Ctrl+Enter to submit)", height=110,
                            value="" if ex == "(write your own)" else ex,
                            key=f"tweet_text_{ex}")
        submitted = st.form_submit_button("Analyze", type="primary")
    if submitted and not text.strip():
        st.warning("Please type a tweet first.")
    if submitted and text.strip():
        probs = predict_proba(model, [text])[0]
        pred_idx = int(np.argmax(probs))
        top_label, confidence = LABELS[pred_idx], float(probs[pred_idx])
        uncertain = confidence < conf_threshold
        label = top_label
        c1, c2 = st.columns([1, 2])
        c1.metric("Predicted sentiment", "Uncertain" if uncertain else top_label)
        c1.caption(f"Top class: {top_label} ({confidence:.0%} confidence)")
        if uncertain:
            c1.warning("The model is not confident. This usually happens with factual or off-topic text: "
                       "it was trained on tweets about games and brands, where 'Neutral' is a loose "
                       "mix of questions, ads and news, not 'factual statements'.")
        c1.caption(f"Cleaned text: `{clean_text(text) or '(empty)'}`")
        fig = px.bar(x=LABELS, y=probs, color=LABELS, color_discrete_map=COLORS,
                     labels={"x": "", "y": "Probability"}, range_y=[0, 1])
        fig.update_layout(showlegend=False, height=280, margin=dict(t=10, b=10))
        c2.plotly_chart(fig, width="stretch")

        infl = word_influence(model, text, pred_idx)
        if not infl.empty:
            st.markdown(f"**Which words pushed toward _{label}_?** (drop in probability when the word is removed)")
            fig2 = px.bar(infl, x="influence", y="word", orientation="h",
                          color="influence", color_continuous_scale="RdYlGn")
            fig2.update_layout(height=max(220, 30 * len(infl)), coloraxis_showscale=False,
                               margin=dict(t=10, b=10))
            st.plotly_chart(fig2, width="stretch")

# ---------- Batch ----------
with tab_batch:
    st.subheader("Score a CSV of tweets")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        df_up = pd.read_csv(up)
        col = st.selectbox("Column that contains the tweet text", df_up.columns,
                           index=int(np.argmax([df_up[c].astype(str).str.len().mean() for c in df_up.columns])))
        if st.button("Run predictions"):
            probs = predict_proba(model, df_up[col].astype(str).tolist())
            out = df_up.copy()
            out["predicted_sentiment"] = [LABELS[i] for i in probs.argmax(1)]
            out["confidence"] = probs.max(1).round(3)
            counts = out["predicted_sentiment"].value_counts().reindex(LABELS, fill_value=0)
            m = st.columns(3)
            for c, l in zip(m, LABELS):
                c.metric(l, f"{counts[l]:,}", f"{counts[l] / len(out):.0%}", delta_color="off")
            fig = px.pie(values=counts.values, names=counts.index, color=counts.index,
                         color_discrete_map=COLORS, hole=0.45)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(out, width="stretch", height=300)
            st.download_button("Download results", out.to_csv(index=False).encode(),
                               "sentiment_predictions.csv", "text/csv")
    else:
        st.info("Any CSV works as long as one column has the tweet text.")

# ---------- Explore ----------
with tab_explore:
    st.subheader("Explore the training data")
    entities = sorted(data["entity"].unique())
    chosen = st.multiselect("Filter by entity (empty = all)", entities)
    view = data[data["entity"].isin(chosen)] if chosen else data

    k = st.columns(4)
    k[0].metric("Tweets", f"{len(view):,}")
    k[1].metric("Entities", view["entity"].nunique())
    k[2].metric("% Positive", f"{(view['sentiment'] == 'Positive').mean():.0%}")
    k[3].metric("% Negative", f"{(view['sentiment'] == 'Negative').mean():.0%}")

    left, right = st.columns(2)
    dist = view["sentiment"].value_counts().reindex(LABELS)
    left.plotly_chart(px.pie(values=dist.values, names=dist.index, color=dist.index,
                             color_discrete_map=COLORS, hole=0.45, title="Sentiment mix"),
                      width="stretch")
    by_ent = (view.groupby(["entity", "sentiment"]).size().unstack(fill_value=0)
              .reindex(columns=LABELS, fill_value=0))
    by_ent = by_ent.div(by_ent.sum(axis=1), axis=0).mul(100).sort_values("Positive")
    right.plotly_chart(px.bar(by_ent, orientation="h", color_discrete_map=COLORS,
                              title="Sentiment % by entity", labels={"value": "%", "entity": ""}),
                       width="stretch")

    st.markdown("#### Most frequent words")
    senti = st.radio("Sentiment", LABELS, index=2, horizontal=True)
    texts = tuple(view.loc[view["sentiment"] == senti, "clean_text"])
    if texts:
        tw = top_words(texts, 50)
        a, b = st.columns(2)
        a.plotly_chart(px.bar(tw.head(20), x="count", y="word", orientation="h",
                              color_discrete_sequence=[COLORS[senti]]).update_yaxes(autorange="reversed"),
                       width="stretch")
        wc = WordCloud(width=600, height=400, background_color="white", max_words=50,
                       colormap="Greens" if senti == "Positive" else "Reds" if senti == "Negative" else "Greys")
        b.image(wc.generate_from_frequencies(dict(zip(tw["word"], tw["count"]))).to_array())

    st.markdown("#### Tweet length by sentiment")
    st.plotly_chart(px.box(view, x="sentiment", y="length", color="sentiment",
                           color_discrete_map=COLORS, category_orders={"sentiment": LABELS}),
                    width="stretch")

# ---------- Model report ----------
with tab_model:
    st.subheader("Model report")
    if metrics is None:
        st.warning("Run `python train.py` to generate reports/metrics.json")
    else:
        st.write(f"**Best model:** {metrics['best_model']}  |  train {metrics['n_train']:,} / "
                 f"test {metrics['n_test']:,} tweets")
        comp = pd.DataFrame(metrics["model_comparison"]).T[["accuracy", "macro_f1"]].round(3)
        st.dataframe(comp, width="stretch")
        pc = pd.DataFrame(metrics["per_class"]).T.loc[LABELS, ["precision", "recall", "f1-score"]].round(3)
        a, b = st.columns(2)
        a.markdown("**Per-class metrics**")
        a.dataframe(pc, width="stretch")
        cm = np.array(metrics["confusion_matrix"])
        b.plotly_chart(px.imshow(cm, x=LABELS, y=LABELS, text_auto=True, color_continuous_scale="Blues",
                                 labels=dict(x="Predicted", y="True"), title="Confusion matrix"),
                       width="stretch")
        st.info("The train/test split is grouped by tweet ID. This dataset contains several paraphrases "
                "of each tweet; a plain random split lets them leak into both sets and inflates "
                "accuracy to ~93%. The grouped split gives a realistic estimate.")
