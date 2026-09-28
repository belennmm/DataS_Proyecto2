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

# 3. TRAIN Y TEST 
def evaluate(name, pipe, X_te, y_te, fit_time, cv_auc):
    t0 = time.time()
    proba = pipe.predict_proba(X_te)[:, 1]
    pred_time = (time.time() - t0) / len(X_te) * 1000          # ms por 1 comentario a analizar
    pred = (proba >= 0.5).astype(int)
    row = {"modelo": name, "accuracy": accuracy_score(y_te, pred),
           "precision": precision_score(y_te, pred), "recall": recall_score(y_te, pred),
           "f1": f1_score(y_te, pred), "roc_auc": roc_auc_score(y_te, proba),
           "cv_roc_auc": cv_auc, "tiempo_entrenamiento_s": fit_time,
           "tiempo_prediccion_ms": pred_time}
    return row, proba, pred


def main(use_sbert=False):
    X_tr, X_te, y_tr, y_te, g_tr = get_data()
    print(f"Train: {len(X_tr)} | Holdout: {len(X_te)} | "
          f"% violación train={y_tr.mean():.2%} holdout={y_te.mean():.2%}")
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    rows, probas, preds, rocs = [], {}, {}, {}
    for name, (pipe, grid) in build_models(use_sbert).items():
        print(f"\n=== {name} ===")
        gs = GridSearchCV(pipe, grid, scoring=SCORING, cv=cv, n_jobs=1, refit=True)
        t0 = time.time()
        gs.fit(X_tr, y_tr, groups=g_tr)
        fit_time = time.time() - t0
        print(f"Mejores parámetros: {gs.best_params_} | AUC CV: {gs.best_score_:.4f}")

        pd.DataFrame(gs.cv_results_).sort_values("rank_test_score").to_csv(
            f"resultados/cv_{name.replace(' ', '_').replace('+', '')}.csv", index=False)
        row, proba, pred = evaluate(name, gs.best_estimator_, X_te, y_te, fit_time,
                                    gs.best_score_)
        row["mejores_parametros"] = str(gs.best_params_)
        rows.append(row)
        probas[name], preds[name] = proba, pred
        fpr, tpr, _ = roc_curve(y_te, proba)
        rocs[name] = (fpr, tpr)
        joblib.dump(gs.best_estimator_, f"modelos/{name.replace(' ', '_').replace('+', '')}.joblib")
        print({k: round(v, 4) for k, v in row.items() if isinstance(v, float)})

    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics.to_csv("resultados/metricas.csv", index=False)
    pickle.dump(rocs, open("resultados/roc_data.pkl", "wb"))
    print("\n", metrics.drop(columns="mejores_parametros").round(4).to_string(index=False))

    holdout = X_te[["row_id", "body", "rule_short", "subreddit"]].copy()
    holdout["y_true"] = y_te.values
    for n in probas:
        holdout[f"proba_{n}"] = probas[n]
    holdout.to_csv("resultados/predicciones_holdout.csv", index=False)

    make_figures(metrics, probas, preds, rocs, y_te, X_te)
    write_submission(metrics)


# 4. FIGURAS GENERADAS
def make_figures(metrics, probas, preds, rocs, y_te, X_te):
    order = metrics["modelo"].tolist()
    palette = dict(zip(order, sns.color_palette("Set2", len(order))))

    # Comparación de métricas
    m = metrics.melt(id_vars="modelo", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"])
    plt.figure(figsize=(11, 5))
    sns.barplot(data=m, x="variable", y="value", hue="modelo", hue_order=order, palette=palette)
    plt.ylim(0.4, 1); plt.title("Comparación de métricas en el conjunto de prueba")
    plt.xlabel(""); plt.ylabel("Valor"); plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout(); plt.savefig("figuras/modelos_01_metricas.png", dpi=150); plt.close()

    # Curvas ROC
    plt.figure(figsize=(6.5, 6))
    for n in order:
        auc = metrics.set_index("modelo").loc[n, "roc_auc"]
        plt.plot(*rocs[n], label=f"{n} (AUC={auc:.3f})", color=palette[n], lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos"); plt.title("Curvas ROC")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig("figuras/modelos_02_roc.png", dpi=150); plt.close()

    # Matrices de confusión
    fig, axes = plt.subplots(1, len(order), figsize=(3.6 * len(order), 3.6))
    for ax, n in zip(np.atleast_1d(axes), order):
        sns.heatmap(confusion_matrix(y_te, preds[n]), annot=True, fmt="d", cmap="Blues",
                    cbar=False, ax=ax, xticklabels=["No viola", "Viola"],
                    yticklabels=["No viola", "Viola"])
        ax.set_title(n, fontsize=10); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    plt.tight_layout(); plt.savefig("figuras/modelos_03_confusion.png", dpi=150); plt.close()

    # Eficiencia: tiempo de entrenamiento vs AUC
    plt.figure(figsize=(7, 5))
    for _, r in metrics.iterrows():
        plt.scatter(r["tiempo_entrenamiento_s"], r["roc_auc"], s=140, color=palette[r["modelo"]])
        plt.annotate(r["modelo"], (r["tiempo_entrenamiento_s"], r["roc_auc"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=9)
    plt.xscale("log"); plt.xlabel("Tiempo de entrenamiento + tuneo (s, escala log)")
    plt.ylabel("ROC-AUC (holdout)"); plt.title("Efectividad vs tiempo de procesamiento")
    plt.tight_layout(); plt.savefig("figuras/modelos_04_tiempo_vs_auc.png", dpi=150); plt.close()

    # AUC por regla (¿el modelo rinde igual en ambas reglas?)
    recs = []
    for n in order:
        for rule in X_te["rule_short"].unique():
            mask = (X_te["rule_short"] == rule).values
            recs.append({"modelo": n, "regla": {"ad": "No Advertising", "legal": "No legal advice"}[rule],
                         "roc_auc": roc_auc_score(y_te[mask], probas[n][mask])})
    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=pd.DataFrame(recs), x="modelo", y="roc_auc", hue="regla", palette="Set1")
    plt.ylim(0.4, 1); plt.xticks(rotation=15); plt.title("ROC-AUC por regla evaluada")
    plt.tight_layout(); plt.savefig("figuras/modelos_05_auc_por_regla.png", dpi=150); plt.close()

    # Interpretabilidad: palabras más influyentes en la Regresión Logística
    try:
        pipe = joblib.load("modelos/Regresión_Logística.joblib")
        names = pipe.named_steps["feats"].get_feature_names_out()
        coef = pd.Series(pipe.named_steps["clf"].coef_[0], index=names)
        coef = coef[coef.index.str.startswith("tfidf__")]
        coef.index = coef.index.str.replace("tfidf__", "")
        top = pd.concat([coef.nsmallest(15), coef.nlargest(15)]).sort_values()
        plt.figure(figsize=(8, 8))
        top.plot(kind="barh", color=["#4C72B0" if v < 0 else "#C44E52" for v in top])
        plt.title("Términos más influyentes (Reg. Logística)\nrojo = empuja a 'viola', azul = 'no viola'\n"
                  "(prefijo ad_/legal_ = término específico de esa regla)")
        plt.tight_layout(); plt.savefig("figuras/modelos_06_terminos_influyentes.png", dpi=150)
        plt.close()
    except Exception as e:
        print("No se pudo generar la figura de términos:", e)


# 5. SUBMISSION
def write_submission(metrics):
    best = metrics.iloc[0]["modelo"]
    pipe = joblib.load(f"modelos/{best.replace(' ', '_').replace('+', '')}.joblib")
    test = preprocess_dataframe(pd.read_csv("Datos/test.csv").drop_duplicates())
    pd.DataFrame({"row_id": test["row_id"],
                  "rule_violation": pipe.predict_proba(test)[:, 1]}
                 ).to_csv("resultados/submission.csv", index=False)
    print(f"\nMejor modelo por AUC: {best} -> resultados/submission.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sbert", action="store_true", help="incluir SBERT + LR (requiere sentence-transformers)")
    main(ap.parse_args().sbert)