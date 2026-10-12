"""
modelos.py  -  Jigsaw Agile Community Rules (clasificación binaria: rule_violation)
Uso:   python modelos.py            (usa Datos/train.csv y Datos/test.csv)
       python modelos.py --sbert    (agrega el modelo con embeddings Transformer)

GPU: si hay una GPU NVIDIA con CUDA se usa automáticamente en XGBoost y en SBERT
(los modelos de scikit-learn -LR, NB, SVM, RF- solo corren en CPU).
No se guardan checkpoints durante el entrenamiento: todo se exporta al final.

Genera:
  modelos/*.joblib                    pipelines completos (preprocesado + modelo) para la app
  modelos/modelo_final.joblib/.json   mejor modelo según AUC de validación cruzada
  resultados/metricas.csv             métricas en el conjunto de prueba (holdout)
  resultados/cv_<modelo>.csv          resultados del ajuste de hiperparámetros
  resultados/roc_data.pkl             curvas ROC
  resultados/predicciones_holdout.csv predicciones por fila (análisis de errores)
  figuras/modelos_*.png               gráficas estáticas para el informe
  resultados/submission.csv           predicciones sobre test.csv
"""

import argparse
import json
import os
import pickle
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
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

import paleta as PAL
from preprocesamiento import (BINARY_COLS, NUMERIC_COLS, TARGET, SentenceEmbedder,
                              gpu_disponible, limpiar_cache_embeddings, load_train,
                              preprocess_dataframe)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

SEED = 42
USE_SUBREDDIT = True
TEXT_COL = "text_rule"      # body_lemmatized + prefijo de regla (ver variantes.py)
SBERT_TEXT_COL = "body"     # los transformers usan texto natural, nunca lematizado
N_SPLITS = 5
SCORING = "roc_auc"    # en Kaggle dice que la competencia evalúa con AUC; se tunea con la misma métrica
DEVICE = "cuda" if gpu_disponible() else "cpu"
GPU_MODELS = {"XGBoost", "SBERT + Reg. Logística"}   # se tunean con n_jobs=1 para no saturar la GPU
for d in ["modelos", "resultados", "figuras"]:
    os.makedirs(d, exist_ok=True)


def file_name(name):
    return name.replace(" ", "_").replace("+", "")


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
def make_features(scaler, max_features=8000, ngram=(1, 2), text_col=TEXT_COL, solo_texto=False):
    transformers = [
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram,
                                  min_df=2, sublinear_tf=True), text_col),
    ]
    if not solo_texto:
        transformers += [
            ("num", scaler, NUMERIC_COLS + BINARY_COLS),
            ("rule", OneHotEncoder(handle_unknown="ignore"), ["rule_short"]),
        ]
        if USE_SUBREDDIT:
            transformers.append(("sub", OneHotEncoder(handle_unknown="ignore", min_frequency=3),
                                 ["subreddit"]))
    return ColumnTransformer(transformers)


def make_sbert_features(text_col=SBERT_TEXT_COL):
    return ColumnTransformer([
        ("emb", SentenceEmbedder(), text_col),
        ("num", StandardScaler(), NUMERIC_COLS + BINARY_COLS),
        ("rule", OneHotEncoder(handle_unknown="ignore"), ["rule_short"]),
        ("sub", OneHotEncoder(handle_unknown="ignore", min_frequency=3), ["subreddit"]),
    ])


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
                                            tree_method="hist", device=DEVICE, n_jobs=-1))]),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 6],
             "clf__learning_rate": [0.05, 0.1]})
    except ImportError:
        print("xgboost no instalado, use `pip install xgboost` ")
    if use_sbert:  # modelo deep learning (opcional como parámetro al momento de ejecutar): embeddings Transformer + clasificador lineal
        models["SBERT + Reg. Logística"] = (
            Pipeline([("feats", make_sbert_features()),
                      ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
            {"clf__C": [0.1, 1, 10]})
    return models

# 3. TRAIN Y TEST
def evaluate(name, pipe, X_te, y_te, fit_time, tune_time, cv_auc):
    t0 = time.time()
    proba = pipe.predict_proba(X_te)[:, 1]
    pred_time = (time.time() - t0) / len(X_te) * 1000          # ms por 1 comentario a analizar
    pred = (proba >= 0.5).astype(int)
    row = {"modelo": name, "accuracy": accuracy_score(y_te, pred),
           "precision": precision_score(y_te, pred), "recall": recall_score(y_te, pred),
           "f1": f1_score(y_te, pred), "roc_auc": roc_auc_score(y_te, proba),
           "cv_roc_auc": cv_auc,
           "tiempo_entrenamiento_s": fit_time,      # un solo ajuste con los mejores parámetros (comparable)
           "tiempo_tuneo_s": tune_time,             # búsqueda completa (depende del tamaño del grid)
           "tiempo_prediccion_ms": pred_time}
    return row, proba, pred


def main(use_sbert=False):
    X_tr, X_te, y_tr, y_te, g_tr = get_data()
    print(f"Dispositivo para XGBoost/SBERT: {DEVICE}")
    print(f"Train: {len(X_tr)} | Holdout: {len(X_te)} | "
          f"% violación train={y_tr.mean():.2%} holdout={y_te.mean():.2%}")
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    rows, probas, preds, rocs, trained, cv_tables = [], {}, {}, {}, {}, {}
    for name, (pipe, grid) in build_models(use_sbert).items():
        print(f"\n=== {name} ===")
        n_jobs = 1 if name in GPU_MODELS else -1
        gs = GridSearchCV(pipe, grid, scoring=SCORING, cv=cv, n_jobs=n_jobs, refit=False)
        t0 = time.time()
        gs.fit(X_tr, y_tr, groups=g_tr)
        tune_time = time.time() - t0
        print(f"Mejores parámetros: {gs.best_params_} | AUC CV: {gs.best_score_:.4f}")

        # ajuste final con los mejores parámetros (cronometrado aparte para comparar modelos;
        # se vacía la cache de embeddings para que SBERT cuente el costo real de codificar)
        limpiar_cache_embeddings()
        final = clone(pipe).set_params(**gs.best_params_)
        t0 = time.time()
        final.fit(X_tr, y_tr)
        fit_time = time.time() - t0

        cv_tables[name] = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
        row, proba, pred = evaluate(name, final, X_te, y_te, fit_time, tune_time,
                                    gs.best_score_)
        row["mejores_parametros"] = str(gs.best_params_)
        rows.append(row)
        trained[name] = final
        probas[name], preds[name] = proba, pred
        fpr, tpr, _ = roc_curve(y_te, proba)
        rocs[name] = (fpr, tpr)
        print({k: round(v, 4) for k, v in row.items() if isinstance(v, float)})

    # el modelo se selecciona con el AUC de validación cruzada; el holdout queda solo para reportar
    metrics = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    print("\n", metrics.drop(columns="mejores_parametros").round(4).to_string(index=False))

    holdout = X_te[["row_id", "body", "rule_short", "subreddit"]].copy()
    holdout["y_true"] = y_te.values
    for n in probas:
        holdout[f"proba_{n}"] = probas[n]

    # 4. EXPORTACIÓN (solo al terminar todo el entrenamiento)
    metrics.to_csv("resultados/metricas.csv", index=False)
    for name, table in cv_tables.items():
        table.to_csv(f"resultados/cv_{file_name(name)}.csv", index=False)
    with open("resultados/roc_data.pkl", "wb") as f:
        pickle.dump(rocs, f)
    holdout.to_csv("resultados/predicciones_holdout.csv", index=False)
    export_models(trained, metrics)

    make_figures(metrics, probas, preds, rocs, y_te, X_te, trained)
    write_submission(metrics, trained)


def export_models(trained, metrics):
    for name, pipe in trained.items():
        joblib.dump(pipe, f"modelos/{file_name(name)}.joblib")
    best = metrics.iloc[0]
    joblib.dump(trained[best["modelo"]], "modelos/modelo_final.joblib")
    with open("modelos/modelo_final.json", "w", encoding="utf-8") as f:
        json.dump({"modelo": best["modelo"], "mejores_parametros": best["mejores_parametros"],
                   "cv_roc_auc": float(best["cv_roc_auc"]), "holdout_roc_auc": float(best["roc_auc"]),
                   "text_col": SBERT_TEXT_COL if "SBERT" in best["modelo"] else TEXT_COL,
                   "dispositivo_entrenamiento": DEVICE}, f, ensure_ascii=False, indent=2)
    print(f"\nModelos exportados en modelos/ | modelo final: {best['modelo']}")


# 5. FIGURAS GENERADAS
def make_figures(metrics, probas, preds, rocs, y_te, X_te, trained):
    order = metrics["modelo"].tolist()
    palette = {m: PAL.model_color(m) for m in order}   # mismo color por modelo que en la app

    # Comparación de métricas (barras desde 0 para no exagerar las diferencias)
    m = metrics.melt(id_vars="modelo", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"])
    plt.figure(figsize=(11, 5))
    sns.barplot(data=m, x="variable", y="value", hue="modelo", hue_order=order, palette=palette)
    plt.ylim(0, 1); plt.title("Comparación de métricas en el conjunto de prueba")
    plt.xlabel(""); plt.ylabel("Valor"); plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout(); plt.savefig("figuras/modelos_01_metricas.png", dpi=150); plt.close()

    # Curvas ROC
    plt.figure(figsize=(6.5, 6))
    for n in order:
        auc = metrics.set_index("modelo").loc[n, "roc_auc"]
        plt.plot(*rocs[n], label=f"{n} (AUC={auc:.3f})", color=palette[n], lw=2)
    plt.plot([0, 1], [0, 1], "--", color="#C9D1D8", lw=1); plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos"); plt.title("Curvas ROC")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig("figuras/modelos_02_roc.png", dpi=150); plt.close()

    # Matrices de confusión
    from matplotlib.colors import LinearSegmentedColormap
    cmap_seq = LinearSegmentedColormap.from_list("peri", PAL.SEQ_SCALE)   # secuencial de un solo tono
    ncols = min(3, len(order))
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(order):]:
        ax.axis("off")
    for ax, n in zip(axes, order):
        sns.heatmap(confusion_matrix(y_te, preds[n]), annot=True, fmt="d", cmap=cmap_seq,
                    cbar=False, ax=ax, xticklabels=["No viola", "Viola"],
                    yticklabels=["No viola", "Viola"])
        ax.set_title(n, fontsize=10); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    plt.tight_layout(); plt.savefig("figuras/modelos_03_confusion.png", dpi=150); plt.close()

    # Eficiencia: tiempo de entrenamiento vs AUC
    plt.figure(figsize=(7, 5))
    for _, r in metrics.iterrows():
        # un solo color + etiqueta directa: en un scatter cualquier par de puntos puede quedar cerca
        plt.scatter(r["tiempo_entrenamiento_s"], r["roc_auc"], s=140, color=PAL.NEUTRAL, edgecolor="white")
        plt.annotate(r["modelo"], (r["tiempo_entrenamiento_s"], r["roc_auc"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=9)
    plt.xscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    plt.gca().xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    plt.gca().xaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("Tiempo de entrenamiento con los mejores parámetros (s, escala log)")
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
    sns.barplot(data=pd.DataFrame(recs), x="regla", y="roc_auc", hue="modelo", hue_order=order,
                palette=palette)
    plt.ylim(0, 1); plt.xlabel(""); plt.title("ROC-AUC por regla evaluada")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout(); plt.savefig("figuras/modelos_05_auc_por_regla.png", dpi=150); plt.close()

    # Interpretabilidad: palabras más influyentes en la Regresión Logística
    try:
        pipe = trained["Regresión Logística"]
        names = pipe.named_steps["feats"].get_feature_names_out()
        coef = pd.Series(pipe.named_steps["clf"].coef_[0], index=names)
        coef = coef[coef.index.str.startswith("tfidf__")]
        coef.index = coef.index.str.replace("tfidf__", "")
        top = pd.concat([coef.nsmallest(15), coef.nlargest(15)]).sort_values()
        plt.figure(figsize=(8, 8))
        top.plot(kind="barh", color=[PAL.PERI if v < 0 else PAL.ORANGE for v in top])
        plt.title("Términos más influyentes (Reg. Logística)\nnaranja = empuja a 'viola', azul = 'no viola'\n"
                  "(prefijo ad_/legal_ = término específico de esa regla)")
        plt.tight_layout(); plt.savefig("figuras/modelos_06_terminos_influyentes.png", dpi=150)
        plt.close()
    except Exception as e:
        print("No se pudo generar la figura de términos:", e)


# 6. SUBMISSION
def write_submission(metrics, trained):
    # nota: los 10 comentarios de test.csv también aparecen en train.csv, por lo que este
    # archivo solo sirve como formato de entrega, no como evaluación del modelo
    best = metrics.iloc[0]["modelo"]
    test = preprocess_dataframe(pd.read_csv("Datos/test.csv").drop_duplicates())
    pd.DataFrame({"row_id": test["row_id"],
                  "rule_violation": trained[best].predict_proba(test)[:, 1]}
                 ).to_csv("resultados/submission.csv", index=False)
    print(f"Mejor modelo por AUC de CV: {best} -> resultados/submission.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sbert", action="store_true", help="incluir SBERT + LR (requiere sentence-transformers)")
    main(ap.parse_args().sbert)
