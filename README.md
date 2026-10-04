# Tweet Sentiment Analysis + Streamlit Dashboard

Entity-level sentiment classification (Negative / Neutral / Positive) of tweets about games and brands,
with an interactive Streamlit dashboard.

## Features
- **Predict**: type a tweet, get the sentiment, class probabilities and the words that drove the prediction.
- **Batch scoring**: upload a CSV, get predictions for every row, download the results.
- **Explore data**: filter by entity, sentiment mix, top words, word clouds, tweet length.
- **Model report**: model comparison, per-class precision/recall/F1, confusion matrix.

## Project structure
```
sentiment_analysis.ipynb  step-by-step notebook with explanations and outputs
app.py            Streamlit dashboard
train.py          trains + compares models, saves the best pipeline and reports
src/preprocess.py text cleaning (shared by training and app)
src/data.py       data loading / cleaning
data/             twitter_training.csv
models/           saved sklearn pipeline (joblib)
reports/          metrics.json, confusion_matrix.png
```

## Run locally
Start with `sentiment_analysis.ipynb` for the full walkthrough (needs `pip install jupyter`), then launch the dashboard:
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py            # optional: a trained model is already included
streamlit run app.py
```

## Method
1. Drop missing text, exact duplicates and the `Irrelevant` class (as in the original coursework).
2. Clean text: lowercase, strip URLs/@mentions/numbers/symbols, map emoticons to tokens, remove stopwords
   **but keep negations** (`not`, `never`, ...).
3. TF-IDF with unigrams + bigrams.
4. Compare Complement Naive Bayes, Logistic Regression and a calibrated Linear SVM; tune with GridSearchCV on macro-F1.
5. Evaluate on a **group split by tweet ID**.

### Changes from the original notebook
| Original | This version | Why |
|---|---|---|
| Emoticon regex ran after punctuation removal | Run before | The patterns could never match otherwise |
| Stopwords (incl. "not") removed; `stopwords.words()` called per word | Negations kept; stopword set precomputed | Better sentiment signal, much faster |
| Lemmatize + stem, NLTK downloads needed | No NLTK dependency | Simpler to deploy, no accuracy loss with bigrams |
| BoW/TF-IDF unigrams, RBF SVM, dense Keras net | TF-IDF 1-2 grams, linear models | Train in seconds instead of minutes |
| Random train/test split | Group split by tweet ID | See below |
| Notebook only | Scripts + saved pipeline + dashboard | Reproducible and deployable |

### Note on evaluation
The dataset contains several paraphrases of each tweet under the same `tweet_id`. With a plain random split the
same tweet appears in train and test, and a linear model scores about 93% accuracy. Splitting by `tweet_id`
removes that leakage and gives a realistic **~65% accuracy / ~0.65 macro-F1**. Neutral is the hardest class.

## Ideas for next steps
- Fine-tune a transformer (e.g. DistilBERT / RoBERTa-sentiment) and compare against this baseline.
- Use the entity as an extra feature.
- Add error analysis on misclassified tweets.

## Data
Twitter Entity Sentiment Analysis dataset (Kaggle). Check its license before redistributing.
