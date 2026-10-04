"""Text cleaning shared by training and the Streamlit app."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Keep negations: removing "not", "no", "never" flips the meaning of a sentence.
NEGATIONS = {"not", "no", "nor", "never", "cannot", "nothing", "nobody", "none"}
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) - NEGATIONS

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
EMOTICONS = [
    (re.compile(r"<3"), " heart "),
    (re.compile(r"(?<!\w)[8:=;]['`\-]?[)d]+(?=\s|$)", re.I), " smile "),
    (re.compile(r"(?<!\w)[8:=;]['`\-]?\(+(?=\s|$)"), " sadface "),
    (re.compile(r"(?<!\w)[8:=;]['`\-]?p+(?=\s|$)", re.I), " lolface "),
    (re.compile(r"(?<!\w)[8:=;]['`\-]?[/|l*](?=\s|$)", re.I), " neutralface "),
]
NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    """Lowercase, handle URLs/mentions/emoticons, drop numbers/symbols and stopwords."""
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    # Emoticons are converted BEFORE punctuation is stripped. The original notebook did it
    # after, so the emoticon patterns could never match.
    for pattern, repl in EMOTICONS:
        text = pattern.sub(repl, text)
    text = NON_ALPHA_RE.sub(" ", text.lower())
    return " ".join(w for w in text.split() if w not in STOPWORDS and len(w) > 1)
