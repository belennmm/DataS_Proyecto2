"""
variantes.py  -  Experimento: ¿cómo cambia el rendimiento según el preprocesamiento del texto?
Uso:   python variantes.py           (Regresión Logística, rápido, CPU)
       python variantes.py --sbert   (agrega SBERT con texto original vs limpio; usa GPU si hay)

Compara, con la MISMA partición, la MISMA validación cruzada y el MISMO grid que modelos.py:
  - texto sin limpiar / limpio / lematizado
  - con y sin prefijo de regla (ad_/legal_)
  - solo texto vs texto + variables extra (numéricas, regla, subreddit)
Para SBERT solo se compara texto original vs limpio: un transformer se entrena con lenguaje
natural, quitarle stopwords o lematizarlo le quita información.

Además del AUC de CV y del holdout, se repite la validación cruzada agrupada 5 veces con
distintas semillas (5x5) para reportar media ± desviación: con ~2000 filas un solo holdout
es ruidoso y diferencias menores a la desviación no son significativas.

Genera: resultados/variantes.csv, figuras/modelos_07_variantes.png
No exporta modelos (es un experimento de comparación); el modelo final lo exporta modelos.py.
"""

import argparse
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modelos import (DEVICE, N_SPLITS, SCORING, SEED, build_models, get_data,
                     make_features, make_sbert_features)
import paleta as PAL
from preprocesamiento import raw_tokens, rule_aware_text

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

N_REPEATS = 5
TEXTOS = {"Sin limpiar": "body_raw", "Limpio": "body_clean", "Lematizado": "body_lemmatized"}


def add_text_variants(X):
    X = X.copy()
    X["body_raw"] = X["body"].apply(raw_tokens)
    for col in TEXTOS.values():
        X[f"{col}_rule"] = [rule_aware_text(t, r) for t, r in zip(X[col], X["rule_short"])]
    return X


def lr_pipe(text_col, extras):
    return Pipeline([("feats", make_features(StandardScaler(), text_col=text_col,
                                             solo_texto=not extras)),
                     ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])


def sbert_pipe(text_col):
    return Pipeline([("feats", make_sbert_features(text_col)),
                     ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])


def run_variant(pipe, grid, data, n_jobs):
    X_tr, X_te, y_tr, y_te, g_tr, X_all, y_all, g_all = data
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, grid, scoring=SCORING, cv=cv, n_jobs=n_jobs, refit=False)
    gs.fit(X_tr, y_tr, groups=g_tr)
    best = clone(pipe).set_params(**gs.best_params_)
    holdout = roc_auc_score(y_te, best.fit(X_tr, y_tr).predict_proba(X_te)[:, 1])
    rep = []
    for i in range(N_REPEATS):   # CV repetida sobre todos los datos con los mejores parámetros
        cv_i = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=100 + i)
        rep += list(cross_val_score(best, X_all, y_all, groups=g_all, cv=cv_i,
                                    scoring=SCORING, n_jobs=n_jobs))
    return {"cv_auc": gs.best_score_, "holdout_auc": holdout,
            "auc_rep_media": np.mean(rep), "auc_rep_std": np.std(rep),
            "mejores_parametros": str(gs.best_params_)}


def main(use_sbert=False):
    X_tr, X_te, y_tr, y_te, g_tr = get_data()
    X_tr, X_te = add_text_variants(X_tr), add_text_variants(X_te)
    X_all, y_all = pd.concat([X_tr, X_te]), pd.concat([y_tr, y_te])
    data = (X_tr, X_te, y_tr, y_te, g_tr, X_all, y_all, X_all["body"])
    print(f"Dispositivo para SBERT: {DEVICE}")

    lr_grid = build_models()["Regresión Logística"][1]
    rows = []
    for extras in [False, True]:
        for texto, col in TEXTOS.items():
            for prefijo in [False, True]:
                t0 = time.time()
                text_col = f"{col}_rule" if prefijo else col
                r = run_variant(lr_pipe(text_col, extras), lr_grid, data, n_jobs=-1)
                rows.append({"modelo": "Regresión Logística", "texto": texto,
                             "prefijo_regla": prefijo, "features_extra": extras, **r,
                             "tiempo_s": time.time() - t0})
                print(f"LR | {texto:11s} | prefijo={prefijo!s:5s} | extras={extras!s:5s} | "
                      f"CV={r['cv_auc']:.4f} holdout={r['holdout_auc']:.4f} "
                      f"5x5={r['auc_rep_media']:.4f}±{r['auc_rep_std']:.4f}")

    if use_sbert:
        for texto, col in [("SBERT original", "body"), ("SBERT limpio", "body_clean")]:
            t0 = time.time()
            r = run_variant(sbert_pipe(col), {"clf__C": [0.1, 1, 10]}, data, n_jobs=1)
            rows.append({"modelo": "SBERT + Reg. Logística", "texto": texto,
                         "prefijo_regla": False, "features_extra": True, **r,
                         "tiempo_s": time.time() - t0})
            print(f"SBERT | {texto} | CV={r['cv_auc']:.4f} holdout={r['holdout_auc']:.4f} "
                  f"5x5={r['auc_rep_media']:.4f}±{r['auc_rep_std']:.4f}")

    res = pd.DataFrame(rows).sort_values("auc_rep_media", ascending=False)
    res.to_csv("resultados/variantes.csv", index=False)
    print("\n", res.drop(columns="mejores_parametros").round(4).to_string(index=False))
    make_figure(res)


def make_figure(res):
    res = res.sort_values("auc_rep_media")
    labels = [f"{r.texto}" + (" + prefijo" if r.prefijo_regla else "")
              + (" + extras" if r.features_extra else " (solo texto)") for r in res.itertuples()]
    # puntos + barras de error (no barras): el eje no empieza en 0 y unas barras recortadas
    # exagerarían visualmente las diferencias
    plt.figure(figsize=(9, 0.45 * len(res) + 1.5))
    for i, (r, label) in enumerate(zip(res.itertuples(), labels)):
        plt.errorbar(r.auc_rep_media, i, xerr=r.auc_rep_std, fmt="o", ms=9, capsize=4,
                     color=PAL.model_color(r.modelo), ecolor="#B7C1C7", label=r.modelo)
    plt.yticks(range(len(res)), labels)
    handles, names = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(names, handles))   # color = modelo, igual que en la app
    plt.legend(uniq.values(), uniq.keys(), title="Modelo", loc="lower right")
    plt.xlabel(f"ROC-AUC (media ± desv. de CV agrupada {N_REPEATS}x{N_SPLITS})")
    plt.title("Efecto del preprocesamiento del texto en el rendimiento")
    plt.tight_layout(); plt.savefig("figuras/modelos_07_variantes.png", dpi=150); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sbert", action="store_true", help="incluir variantes SBERT (requiere sentence-transformers)")
    main(ap.parse_args().sbert)
