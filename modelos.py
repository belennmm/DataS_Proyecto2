"""
modelos.py  -  Jigsaw Agile Community Rules (clasificación binaria: rule_violation)
Uso:   python modelos.py            (usa Datos/train.csv y Datos/test.csv)
       python modelos.py --sbert    (agrega el modelo con embeddings Transformer)

Genera:
  modelos/*.joblib                    pipelines completos (preprocesado + modelo) para la app
  resultados/metricas.csv             métricas en el conjunto de prueba (holdout)
  resultados/cv_<modelo>.csv          resultados del ajuste de hiperparámetros
  resultados/roc_data.pkl             curvas ROC
  resultados/predicciones_holdout.csv predicciones por fila (análisis de errores)
  figuras/modelos_*.png               gráficas estáticas para el informe
  resultados/submission.csv           predicciones sobre test.csv
"""

import argparse
import os
import pickle
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from preprocesamiento import (BINARY_COLS, NUMERIC_COLS, TARGET, SentenceEmbedder,
                              load_train, preprocess_dataframe)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

SEED = 42
USE_SUBREDDIT = True   
TEXT_COL = "text_rule"  
N_SPLITS = 5
SCORING = "roc_auc"    # en Kaggle dice que la competencia evalúa con AUC; se tunea con la misma métrica
for d in ["modelos", "resultados", "figuras"]:
    os.makedirs(d, exist_ok=True)


# 1. DATOS Y PARTICIÓN
def get_data():
    df = load_train("Datos/train.csv")
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    # Si un mismo 'body' aparece varias veces, si cae parte en train y
    # parte en test, hay data leakage, entonces se particiona agrupando por body
    groups = df["body"]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, te_idx = next(sgkf.split(X, y, groups))   # ~80% train / 20% holdout
    return (X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx],
            groups.iloc[tr_idx])


# 2. FEATURES + MODELOS
def make_features(scaler, max_features=8000, ngram=(1, 2)):
    transformers = [
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram,
                                  min_df=2, sublinear_tf=True), TEXT_COL),
        ("num", scaler, NUMERIC_COLS + BINARY_COLS),
        ("rule", OneHotEncoder(handle_unknown="ignore"), ["rule_short"]),
    ]
    if USE_SUBREDDIT:
        transformers.append(("sub", OneHotEncoder(handle_unknown="ignore", min_frequency=3),
                             ["subreddit"]))
    return ColumnTransformer(transformers)


def build_models(use_sbert=False):
    """Devuelve {nombre: (pipeline, grid de hiperparámetros)}."""
    models = {
        "Regresión Logística": (
            Pipeline([("feats", make_features(StandardScaler())),
                      ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
            {"feats__tfidf__ngram_range": [(1, 1), (1, 2)],
             "clf__C": [0.1, 1, 10], "clf__class_weight": [None, "balanced"]}),
        "Naive Bayes": (
            Pipeline([("feats", make_features(MinMaxScaler())),   # NB exige valores >= 0
                      ("clf", MultinomialNB())]),
            {"feats__tfidf__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.01, 0.1, 0.5, 1]}),
        "SVM": (
            Pipeline([("feats", make_features(StandardScaler())),
                      ("clf", SVC(probability=True, random_state=SEED))]),
            {"clf__kernel": ["linear", "rbf"], "clf__C": [0.1, 1, 10]}),
        "Random Forest": (
            Pipeline([("feats", make_features(StandardScaler(), max_features=3000)),
                      ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 20],
             "clf__min_samples_leaf": [1, 3]}),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = (
            Pipeline([("feats", make_features(StandardScaler(), max_features=3000)),
                      ("clf", XGBClassifier(eval_metric="logloss", random_state=SEED,
                                            n_jobs=-1))]),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 6],
             "clf__learning_rate": [0.05, 0.1]})
    except ImportError:
        print("xgboost no instalado, use `pip install xgboost` ")
    if use_sbert:  # modelo deep learning (opcional como parámetro al momento de ejecutar): embeddings Transformer + clasificador lineal
        feats = ColumnTransformer([
            ("emb", SentenceEmbedder(), "body"),
            ("num", StandardScaler(), NUMERIC_COLS + BINARY_COLS),
            ("rule", OneHotEncoder(handle_unknown="ignore"), ["rule_short"]),
            ("sub", OneHotEncoder(handle_unknown="ignore", min_frequency=3), ["subreddit"]),
        ])
        models["SBERT + Reg. Logística"] = (
            Pipeline([("feats", feats),
                      ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
            {"clf__C": [0.1, 1, 10]})
    return models