"""
preprocesamiento.py
Limpieza y creación de variables. Misma lógica del notebook de EDA, modularizada para su reutilización
"""
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

for _p in ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    nltk.download(_p, quiet=True)

TARGET = "rule_violation"
NUMERIC_COLS = ["char_length", "word_count", "url_count", "exclamation_count",
                "emoji_count", "caps_ratio"]
BINARY_COLS = ["has_url", "has_spam_keyword", "has_legal_keyword"]

#  regex / listas
URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)")
MD_LINK_REGEX = re.compile(r"\[([^\]]*)\]\((https?://[^\)]+)\)")
MD_BOLD_ITALIC_REGEX = re.compile(r"(\*\*|\*|__|_)")
MD_QUOTE_REGEX = re.compile(r"^>.*$", flags=re.MULTILINE)
EMOJI_REGEX = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]+",
    flags=re.UNICODE)
NON_ALPHA_REGEX = re.compile(r"[^a-z\s]")

SPAM_KEYWORDS = ["click here", "discount", "free", "subscribe", "buy now", "sale",
                 "offer", "% off", "limited time", "download", "stream", "promo",
                 "referral"]
LEGAL_KEYWORDS = ["lawyer", "sue", "court", "legal", "attorney", "lawsuit", "law ",
                  "illegal", "sued"]

STOPWORDS_EN = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# funciones base
def count_urls(t):
    return len(URL_REGEX.findall(t)) + len(MD_LINK_REGEX.findall(t))

def caps_ratio(t):
    letters = [c for c in t if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0

def clean_text(text):
    text = str(text)
    text = MD_LINK_REGEX.sub(" ", text)
    text = URL_REGEX.sub(" ", text)
    text = MD_QUOTE_REGEX.sub(" ", text)
    text = MD_BOLD_ITALIC_REGEX.sub(" ", text)
    text = text.lower()
    text = EMOJI_REGEX.sub(" ", text)
    text = NON_ALPHA_REGEX.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize_lemmatize(text):
    tokens = [t for t in word_tokenize(text) if t not in STOPWORDS_EN and len(t) > 1]
    return " ".join(LEMMATIZER.lemmatize(t) for t in tokens)

def rule_short(rule):
    return "ad" if "advertising" in str(rule).lower() else "legal"

def rule_aware_text(clean, tag):
    """Palabras normales + palabras prefijadas con la regla (ej. 'legal_sue').
    Hace que un modelo lineal aprenda que una palabra pesa distinto según la regla
    que se evalúa (la etiqueta depende del par comentario-regla, no solo del texto)"""
    words = clean.split()
    return " ".join(words + [f"{tag}_{w}" for w in words])

# pipeline de datos
def preprocess_dataframe(df):
    """Recibe columnas body, rule, subreddit (+ target opcional) y devuelve el df
    con todas las variables derivadas que usan los modelos."""
    df = df.copy()
    b = df["body"].astype(str)
    df["char_length"] = b.str.len()
    df["word_count"] = b.str.split().apply(len)
    df["url_count"] = b.apply(count_urls)
    df["exclamation_count"] = b.str.count("!")
    df["emoji_count"] = b.apply(lambda t: len(EMOJI_REGEX.findall(t)))
    df["caps_ratio"] = b.apply(caps_ratio)
    df["has_url"] = (df["url_count"] > 0).astype(int)
    low = b.str.lower()
    df["has_spam_keyword"] = low.apply(lambda t: int(any(k in t for k in SPAM_KEYWORDS)))
    df["has_legal_keyword"] = low.apply(lambda t: int(any(k in t for k in LEGAL_KEYWORDS)))
    df["body_clean"] = b.apply(clean_text)
    df["body_lemmatized"] = df["body_clean"].apply(tokenize_lemmatize)
    df["rule_short"] = df["rule"].apply(rule_short)
    df["text_rule"] = [rule_aware_text(c, r) for c, r in zip(df["body_clean"], df["rule_short"])]
    return df


def preprocess_input(body, rule, subreddit):
    """Preprocesamiento de una entrada (escalable para la aplicacion)"""
    return preprocess_dataframe(pd.DataFrame(
        {"body": [body], "rule": [rule], "subreddit": [subreddit]}))


def load_train(path="Datos/train.csv"):
    """Carga train, elimina duplicados de FILA COMPLETA (igual que el EDA) y preprocesa."""
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    return preprocess_dataframe(df)


# deep learning ligero
class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """Convierte texto en embeddings con un Transformer preentrenado (SBERT).
    Se usa como 'feature extractor' dentro de un Pipeline de scikit-learn. Opcional"""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not hasattr(self, "_model"):
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        texts = pd.Series(X).astype(str).tolist() if not isinstance(X, pd.DataFrame) \
            else X.iloc[:, 0].astype(str).tolist()
        return np.asarray(self._model.encode(texts, batch_size=64, show_progress_bar=False))

    def __getstate__(self):  # no serializar el modelo pesado; se recarga al usarlo
        state = self.__dict__.copy()
        state.pop("_model", None)
        return state