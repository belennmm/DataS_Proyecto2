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
