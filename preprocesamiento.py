"""
preprocesamiento.py
Limpieza y creación de variables. Misma lógica del notebook de EDA, modularizada para su reutilización
"""
import re
import shutil
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

for _res, _p in [("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet"),
                  ("corpora/omw-1.4", "omw-1.4"), ("tokenizers/punkt", "punkt"),
                  ("tokenizers/punkt_tab", "punkt_tab")]:
    try:   # solo se descarga si falta (arranque rápido y funciona sin internet)
        nltk.data.find(_res)
    except LookupError:
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
                 "promotion", "promotional", "referral"]
LEGAL_KEYWORDS = ["lawyer", "sue", "suing", "court", "legal", "attorney", "lawsuit",
                  "law", "illegal", "sued"]


def keyword_regex(words):
    """Coincidencia por palabra completa (+ flexiones simples: s, es, d, ed, ing, ly).
    Evita falsos positivos por subcadena: 'sue' ya no coincide con 'issue'/'pursue'
    ni 'free' con 'freedom', pero 'lawyers', 'legally' o 'streaming' sí cuentan."""
    parts = []
    for w in words:
        start = r"(?<!\w)" if w[0].isalnum() else ""
        end = r"(?:s|es|d|ed|ing|ly)?(?!\w)" if w[-1].isalnum() else ""
        parts.append(start + re.escape(w) + end)
    return re.compile("|".join(parts))


SPAM_REGEX = keyword_regex(SPAM_KEYWORDS)
LEGAL_REGEX = keyword_regex(LEGAL_KEYWORDS)

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

def raw_tokens(text):
    """Texto SIN limpiar: solo minúsculas y separación en tokens alfanuméricos.
    Conserva URLs/dominios, números y stopwords (se usa en el experimento de variantes)."""
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))

def rule_short(rule):
    return "ad" if "advertising" in str(rule).lower() else "legal"

def rule_aware_text(text, tag):
    """Palabras normales + palabras prefijadas con la regla (ej. 'legal_sue').
    Hace que un modelo lineal aprenda que una palabra pesa distinto según la regla
    que se evalúa (la etiqueta depende del par comentario-regla, no solo del texto)"""
    words = text.split()
    return " ".join(words + [f"{tag}_{w}" for w in words])

# pipeline de datos
def preprocess_dataframe(df):
    """Recibe columnas body, rule, subreddit (+ target opcional) y devuelve el df
    con todas las variables derivadas que usan los modelos."""
    df = df.copy()
    # git en Windows (core.autocrlf) convierte los saltos de línea del CSV a \r\n;
    # se normalizan a \n para que char_length no dependa del sistema operativo
    df["body"] = df["body"].astype(str).str.replace("\r\n", "\n", regex=False)
    b = df["body"]
    df["char_length"] = b.str.len()
    df["word_count"] = b.str.split().apply(len)
    df["url_count"] = b.apply(count_urls)
    df["exclamation_count"] = b.str.count("!")
    df["emoji_count"] = b.apply(lambda t: len(EMOJI_REGEX.findall(t)))
    df["caps_ratio"] = b.apply(caps_ratio)
    df["has_url"] = (df["url_count"] > 0).astype(int)
    low = b.str.lower()
    df["has_spam_keyword"] = low.apply(lambda t: int(bool(SPAM_REGEX.search(t))))
    df["has_legal_keyword"] = low.apply(lambda t: int(bool(LEGAL_REGEX.search(t))))
    df["body_clean"] = b.apply(clean_text)
    df["body_lemmatized"] = df["body_clean"].apply(tokenize_lemmatize)
    df["rule_short"] = df["rule"].apply(rule_short)
    # texto del modelo final: lematizado + prefijo de regla (elegido con variantes.py)
    df["text_rule"] = [rule_aware_text(c, r) for c, r in zip(df["body_lemmatized"], df["rule_short"])]
    return df


def preprocess_input(body, rule, subreddit):
    """Preprocesamiento de una entrada (escalable para la aplicacion)"""
    return preprocess_dataframe(pd.DataFrame(
        {"body": [body], "rule": [rule], "subreddit": [subreddit]}))


def load_train(path="Datos/train.csv"):
    """Carga train, elimina duplicados de FILA COMPLETA (igual que el EDA) y preprocesa."""
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    return preprocess_dataframe(df)


# GPU
def gpu_disponible():
    """True si hay una GPU NVIDIA utilizable (CUDA)."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return shutil.which("nvidia-smi") is not None


# deep learning ligero
_EMB_CACHE = {}   # cache EN MEMORIA (no se guarda en disco): evita recalcular embeddings en cada fold


def limpiar_cache_embeddings():
    _EMB_CACHE.clear()


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """Convierte texto en embeddings con un Transformer preentrenado (SBERT).
    Se usa como 'feature extractor' dentro de un Pipeline de scikit-learn. Opcional.
    device=None -> usa GPU (cuda) si está disponible, si no CPU."""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None,
                 batch_size=64):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def _get_model(self):
        if not hasattr(self, "_model"):
            from sentence_transformers import SentenceTransformer
            device = self.device or ("cuda" if gpu_disponible() else "cpu")
            try:   # primero la copia local (más rápido y funciona sin internet)
                self._model = SentenceTransformer(self.model_name, device=device,
                                                  local_files_only=True)
            except Exception:   # primera vez: se descarga de Hugging Face
                self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def transform(self, X):
        texts = pd.Series(X).astype(str).tolist() if not isinstance(X, pd.DataFrame) \
            else X.iloc[:, 0].astype(str).tolist()
        missing = [t for t in dict.fromkeys(texts) if (self.model_name, t) not in _EMB_CACHE]
        if missing:
            embs = self._get_model().encode(missing, batch_size=self.batch_size,
                                            show_progress_bar=False, convert_to_numpy=True)
            for t, e in zip(missing, embs):
                _EMB_CACHE[(self.model_name, t)] = e
        return np.vstack([_EMB_CACHE[(self.model_name, t)] for t in texts])

    def __getstate__(self):  # no serializar el modelo pesado; se recarga al usarlo
        state = self.__dict__.copy()
        state.pop("_model", None)
        return state
