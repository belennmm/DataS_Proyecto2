"""
app.py  -  r/ModeraciónIA: aplicación interactiva del Proyecto 2
(Jigsaw - Agile Community Rules Classification)

Ejecutar:   streamlit run app.py

Pestañas:
  Inicio       la historia de los datos y de los modelos en forma de posts
  Explorar     variables del dataset limpio (filtros, % / conteos, tablas)
  Clasificar   un comentario nuevo con uno o todos los modelos entrenados
  Rendimiento  métricas interactivas en el holdout (se puede ocultar)
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

import app_estilo as E
from preprocesamiento import BINARY_COLS, NUMERIC_COLS, TARGET, preprocess_input

st.set_page_config(page_title="r/ModeraciónIA · Proyecto 2", page_icon=":material/shield:",
                   layout="wide", initial_sidebar_state="collapsed")

# modo claro / oscuro: cambia los tokens de la app y el tema nativo de Streamlit
st.session_state.setdefault("oscuro", False)
if E.set_modo(st.session_state["oscuro"]):
    st.rerun()   # el navegador recibe el tema nativo al inicio de cada ejecución
E.inject_css()


def cambiar_modo():
    st.session_state["oscuro"] = not st.session_state["oscuro"]

BASE = os.path.dirname(os.path.abspath(__file__))
REGLA_CORTA = {"ad": "No Advertising", "legal": "No legal advice"}
NOMBRES_VAR = {
    "char_length": "Longitud (caracteres)", "word_count": "Cantidad de palabras",
    "url_count": "Cantidad de URLs", "exclamation_count": "Signos de exclamación",
    "emoji_count": "Emojis", "caps_ratio": "Proporción de mayúsculas",
    "has_url": "Tiene URL", "has_spam_keyword": "Palabra clave de spam",
    "has_legal_keyword": "Palabra clave legal", TARGET: "Viola la regla",
}
EJEMPLOS = {   # comentarios reales del conjunto de prueba (holdout)
    "Spam de regalos": ("GlobalOffensive", "No Advertising",
                          "I just found tons of free Steam Wallet Codes at http://steamkeygiveaway.com/ "
                          "You can choose any game you want for free!"),
    "Link de streaming": ("soccerstreams", "No Advertising",
                            "SD Stream : [ English ](http://www.g-livesports.com/)  Mobile Stream : "
                            "[ English ](http://www.g-livesports.com/)"),
    "Consejo legal": ("legaladvice", "No legal advice",
                         "It's illegal, you can sue for pain and suffering from the break in the "
                         "confidentially agreement. I would sue for the work you missed."),
    "Opinión": ("TwoXChromosomes", "No legal advice",
                  "That is called battery.  Two wrongs don't make a right."),
}


def ruta(*p):
    return os.path.join(BASE, *p)


def file_name(name):   # misma convención que modelos.py
    return name.replace(" ", "_").replace("+", "")


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def mostrar(fig, key, height=380, legend=True):
    st.plotly_chart(E.estilo_fig(fig, height, legend), config=E.PLOTLY_CONFIG, key=key, theme=None)


def tabla(df, label="Ver datos de la gráfica", **kw):
    with st.expander(label):
        st.dataframe(df, hide_index=True, **kw)


# =====================================================================
# CARGA DE DATOS Y MODELOS (con cache)
# =====================================================================
@st.cache_data(show_spinner="Cargando datos…")
def cargar_train():
    p = ruta("Datos", "train_clean.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
    else:
        from preprocesamiento import load_train
        df = load_train(ruta("Datos", "train.csv"))
    df["regla"] = np.where(df["rule"].str.contains("Advertising"), "No Advertising", "No legal advice")
    df["clase"] = np.where(df[TARGET] == 1, "Viola", "No viola")
    df["body_lemmatized"] = df["body_lemmatized"].fillna("").astype(str)
    return df


@st.cache_data(show_spinner=False)
def cargar_resultados():
    met = pd.read_csv(ruta("resultados", "metricas.csv"))
    hold = pd.read_csv(ruta("resultados", "predicciones_holdout.csv"))
    hold["regla"] = hold["rule_short"].map(REGLA_CORTA)
    pv = ruta("resultados", "variantes.csv")
    var = pd.read_csv(pv) if os.path.exists(pv) else None
    cv = {}
    for m in met["modelo"]:
        pc = ruta("resultados", f"cv_{file_name(m)}.csv")
        if os.path.exists(pc):
            cv[m] = pd.read_csv(pc)
    pf = ruta("modelos", "modelo_final.json")
    final = json.load(open(pf, encoding="utf-8")) if os.path.exists(pf) else {"modelo": met.iloc[0]["modelo"]}
    return met, hold, var, cv, final


@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_modelo(nombre):
    pipe = joblib.load(ruta("modelos", f"{file_name(nombre)}.joblib"))
    emb = getattr(pipe.named_steps.get("feats"), "named_transformers_", {}).get("emb")
    if emb is not None:   # SBERT: se carga el transformer aquí para no contarlo como tiempo de predicción
        emb._get_model()
    return pipe


train = cargar_train()
met, hold, variantes, cv_tablas, final = cargar_resultados()
MODELOS = met["modelo"].tolist()          # ordenados por AUC de validación cruzada
REGLAS = {r: train.loc[train["regla"] == r, "rule"].iloc[0] for r in ["No Advertising", "No legal advice"]}
mejor = final["modelo"]
fila_mejor = met.set_index("modelo").loc[mejor]

c_top, c_modo = st.columns([12, 1.7], vertical_alignment="center")
with c_top:
    E.topbar("r/ModeraciónIA",
             "¿Este comentario rompe la regla de la comunidad? · Proyecto 2 · CC3084 Data Science · UVG",
             [("Jigsaw · Agile Community Rules", "orange"), ("PLN", "peri"), ("6 modelos", "")])
with c_modo:
    oscuro = st.session_state["oscuro"]
    st.button("Modo claro" if oscuro else "Modo oscuro", key="modo_btn", on_click=cambiar_modo,
              icon=":material/light_mode:" if oscuro else ":material/dark_mode:", width="stretch",
              help="Cambiar entre modo claro y oscuro")
st.write("")

tab_inicio, tab_explorar, tab_clasificar, tab_rend = st.tabs(
    ["Inicio", "Explorar datos", "Clasificar comentario", "Rendimiento de modelos"])


# =====================================================================
# INICIO: la historia
# =====================================================================
with tab_inicio:
    c = st.columns(5)
    with c[0]:
        E.stat("Comentarios evaluados", f"{len(train):,}", "pares (comentario, regla)")
    with c[1]:
        E.stat("Subreddits", f"{train['subreddit'].nunique()}", "comunidades distintas")
    with c[2]:
        E.stat("Reglas", "2", "No Advertising · No legal advice")
    with c[3]:
        E.stat("Violan la regla", pct(train[TARGET].mean()), "dataset balanceado")
    with c[4]:
        E.stat("Mejor modelo", f"AUC {fila_mejor['roc_auc']:.3f}",
               f"{E.MODEL_SHORT.get(mejor, mejor)} · CV {fila_mejor['cv_roc_auc']:.3f}")

    st.write("")
    izq, der = st.columns([1.6, 1], gap="large")
    with izq:
        E.seccion("La historia de los datos", "Los hallazgos más importantes, en formato post")
        leg = train[train["regla"] == "No legal advice"]
        r_kw = leg.groupby("has_legal_keyword")[TARGET].mean()
        E.post(pct(r_kw.get(1, 0), 0), "r/EDA", "orange",
               "Las palabras legales casi garantizan una violación… pero solo bajo la regla legal",
               f"Bajo <b>No legal advice</b>, los comentarios con palabras como <i>lawyer, sue, court</i> "
               f"violan la regla el <b>{pct(r_kw.get(1, 0))}</b> de las veces, contra "
               f"{pct(r_kw.get(0, 0))} sin ellas. Bajo <b>No Advertising</b> la señal desaparece: "
               f"una palabra pesa distinto según la regla que se evalúa.")
        soc = train[train["subreddit"] == "soccerstreams"]
        url_rule = train.groupby(["regla", "has_url"])[TARGET].mean()
        E.post(pct(soc[TARGET].mean(), 0), "r/EDA", "peri",
               "La paradoja de los links: tener URL no significa ser spam",
               f"En r/soccerstreams el <b>{pct(soc['has_url'].mean(), 0)}</b> de los comentarios trae un link, "
               f"pero solo el <b>{pct(soc[TARGET].mean())}</b> viola la regla: ahí compartir streams es lo normal. "
               f"Además, casi todas las URL se evalúan con No Advertising, la regla con menos violaciones; "
               f"dentro de esa regla la diferencia es pequeña ({pct(url_rule[('No Advertising', 1)])} con URL vs "
               f"{pct(url_rule[('No Advertising', 0)])} sin URL).")
        dup = train[train.duplicated("body", keep=False)].groupby("body").agg(
            r=("rule", "nunique"), s=("subreddit", "nunique"), l=(TARGET, "nunique"))
        inc = dup[dup["l"] > 1]
        E.post(f"{len(inc)}/{len(dup)}", "r/Datos", "",
               "Mismo comentario, distinto veredicto",
               f"{len(dup)} textos aparecen más de una vez y {len(inc)} reciben etiquetas distintas. "
               f"Solo {(inc['r'] > 1).sum()} cambian de regla; {((inc['r'] == 1) & (inc['s'] == 1)).sum()} son "
               f"idénticos en texto, regla y subreddit (ruido de etiqueta). Por eso la partición agrupa por "
               f"comentario y ningún modelo puede llegar a un acierto perfecto.")
        lr = met.set_index("modelo").loc["Regresión Logística"] if "Regresión Logística" in MODELOS else None
        texto_mod = (f"<b>{mejor}</b> obtuvo el mejor AUC de validación cruzada "
                     f"(<b>{fila_mejor['cv_roc_auc']:.3f}</b>; {fila_mejor['roc_auc']:.3f} en el holdout).")
        if lr is not None:
            texto_mod += (f" La Regresión Logística con TF-IDF queda muy cerca (CV {lr['cv_roc_auc']:.3f}, holdout "
                          f"{lr['roc_auc']:.3f}) y predice ~{fila_mejor['tiempo_prediccion_ms'] / lr['tiempo_prediccion_ms']:.0f}× "
                          f"más rápido: la diferencia es menor que la variabilidad entre particiones (±0.02).")
        E.post(f"{fila_mejor['cv_roc_auc']:.2f}", "r/Modelos", "orange",
               "Un transformer pequeño gana… por poco", texto_mod)
        E.post("97%", "r/Metodología", "peri",
               "Los ejemplos de referencia filtrarían la respuesta",
               "Los ejemplos positivos/negativos que trae cada fila son, en realidad, comentarios del mismo "
               "train con la misma etiqueta en el 97% de los casos. Usarlos como variables sería fuga de "
               "información, así que los modelos solo usan el comentario, la regla, el subreddit y "
               "variables derivadas del texto.")
    with der:
        E.seccion("Cómo usar la app")
        with st.container(key="card_guia"):
            st.markdown("**1. Explorar datos** — filtra por regla, subreddit y clase; cambia entre "
                        "porcentajes y conteos; haz clic en la leyenda para ocultar series.")
            st.markdown("**2. Clasificar** — escribe un comentario (o carga un ejemplo real), elige la "
                        "regla y uno o todos los modelos, y compara sus respuestas.")
            st.markdown("**3. Rendimiento** — curvas ROC, matrices de confusión, umbral de decisión, "
                        "tiempos y ajuste de hiperparámetros. Se puede ocultar con un interruptor.")
        st.write("")
        E.seccion("Código de colores")
        with st.container(key="card_colores"):
            st.markdown(
                f'<span class="rd-flair orange">● Viola la regla</span>'
                f'<span class="rd-flair peri">● No viola</span>', unsafe_allow_html=True)
            st.caption("Naranja Reddit y azul periwinkle: par cálido/frío que también se distingue con "
                       "daltonismo. Cada modelo conserva su color en todas las gráficas:")
            st.markdown(" ".join(
                f'<span class="rd-flair" style="background:{E.model_color(m)}22;color:{E.INK}">'
                f'<span style="color:{E.model_color(m)}">●</span> {E.esc(m)}</span>' for m in MODELOS),
                unsafe_allow_html=True)


# =====================================================================
# EXPLORAR DATOS
# =====================================================================
with tab_explorar:
    with st.container(key="filtros_explorar"):
        f1, f2, f3, f4 = st.columns([1.6, 1.1, 1.8, 1.5])
        with f1:
            sel_regla = st.pills("Regla", ["No Advertising", "No legal advice"], selection_mode="multi",
                                 default=["No Advertising", "No legal advice"], key="ex_regla")
        with f2:
            sel_clase = st.pills("Clase", E.CLASS_ORDER, selection_mode="multi",
                                 default=E.CLASS_ORDER, key="ex_clase")
        with f3:
            subs_orden = train["subreddit"].value_counts().index.tolist()
            sel_subs = st.multiselect("Subreddits", subs_orden, default=[], key="ex_subs",
                                      placeholder="Todos los subreddits")
        with f4:
            buscar = st.text_input("Buscar en el comentario", key="ex_buscar", placeholder="ej. lawyer, stream…")

    df = train[train["regla"].isin(sel_regla or []) & train["clase"].isin(sel_clase or [])]
    if sel_subs:
        df = df[df["subreddit"].isin(sel_subs)]
    if buscar:
        df = df[df["body"].str.contains(buscar, case=False, regex=False)]
    st.caption(f"**{len(df):,}** comentarios en la selección · {pct(df[TARGET].mean()) if len(df) else '—'} violan la regla. "
               "Los filtros aplican a todas las gráficas de esta pestaña.")

    if len(df) == 0:
        st.warning("No hay comentarios con esos filtros. Prueba quitando alguno.")
    else:
        # ---- 1. distribución de variables ----
        with st.container(key="card_dist"):
            E.seccion("¿Cómo se distribuyen las variables del texto?",
                      "Variables numéricas construidas en el preprocesamiento, separadas por clase")
            a, b, c3 = st.columns([1.4, 1.4, 1.2])
            var = a.selectbox("Variable", NUMERIC_COLS, format_func=NOMBRES_VAR.get, key="ex_var")
            tipo = b.segmented_control("Gráfica", ["Histograma", "Caja", "Violín"], default="Histograma",
                                       key="ex_tipo") or "Histograma"
            unidad = c3.segmented_control("Unidades", ["Conteo", "Porcentaje"], default="Porcentaje",
                                          key="ex_unidad", disabled=tipo != "Histograma") or "Porcentaje"
            if tipo == "Histograma":
                fig = px.histogram(df, x=var, color="clase", barmode="group", nbins=30,
                                   histnorm="percent" if unidad == "Porcentaje" else None,
                                   color_discrete_map=E.CLASS_COLORS,
                                   category_orders={"clase": E.CLASS_ORDER},
                                   labels={var: NOMBRES_VAR[var], "clase": "Clase"})
                fig.update_yaxes(title="% de comentarios de cada clase" if unidad == "Porcentaje" else "Comentarios")
            elif tipo == "Caja":
                fig = px.box(df, x="clase", y=var, color="clase", points="outliers",
                             color_discrete_map=E.CLASS_COLORS, category_orders={"clase": E.CLASS_ORDER},
                             labels={var: NOMBRES_VAR[var], "clase": ""})
            else:
                fig = px.violin(df, x="clase", y=var, color="clase", box=True, points=False,
                                color_discrete_map=E.CLASS_COLORS, category_orders={"clase": E.CLASS_ORDER},
                                labels={var: NOMBRES_VAR[var], "clase": ""})
            mostrar(fig, "fig_dist")
            res = df.groupby("clase")[var].describe().reset_index().round(3)
            medias = df.groupby("clase")[var].mean()
            if {"Viola", "No viola"} <= set(medias.index):
                E.hallazgo(f"En promedio, <b>{NOMBRES_VAR[var].lower()}</b> = {medias['Viola']:.2f} en comentarios "
                           f"que violan la regla vs {medias['No viola']:.2f} en los que no. Los comentarios que violan "
                           f"suelen ser más largos y traer menos URLs; los valores atípicos no se eliminaron porque "
                           f"un texto muy largo o muy corto es una señal legítima.")
            tabla(res)

        st.write("")
        # ---- 2. subreddits ----
        with st.container(key="card_subs"):
            E.seccion("¿En qué comunidades se rompen más las reglas?")
            a, b, c3 = st.columns([1.4, 1.2, 1.2])
            top_n = a.slider("Cantidad de subreddits", 5, 30, 15, key="ex_topn")
            vista = b.segmented_control("Ver como", ["Tasa (%)", "Conteos"], default="Tasa (%)",
                                        key="ex_vista_sub") or "Tasa (%)"
            orden = c3.segmented_control("Ordenar por", ["Frecuencia", "Tasa"], default="Frecuencia",
                                         key="ex_orden_sub") or "Frecuencia"
            g = df.groupby("subreddit").agg(n=(TARGET, "size"), violan=(TARGET, "sum"), tasa=(TARGET, "mean"))
            g["no_violan"] = g["n"] - g["violan"]
            g = g.sort_values("n", ascending=False).head(top_n)
            g = g.sort_values("tasa" if orden == "Tasa" else "n", ascending=True).reset_index()
            if vista == "Tasa (%)":
                fig = go.Figure(go.Bar(
                    y=g["subreddit"], x=100 * g["tasa"], orientation="h", marker_color=E.ORANGE,
                    customdata=g[["n", "violan"]],
                    hovertemplate="<b>%{x:.1f}%</b> violan<br>r/%{y}<br>%{customdata[1]} de %{customdata[0]} comentarios<extra></extra>"))
                fig.add_vline(x=100 * df[TARGET].mean(), line_dash="dot", line_color=E.MUTED,
                              annotation_text=f"promedio {pct(df[TARGET].mean())}",
                              annotation_font_color=E.MUTED, annotation_position="top")
                fig.update_xaxes(title="% de comentarios que violan la regla", range=[0, 100])
                mostrar(fig, "fig_subs", height=max(320, 26 * len(g) + 80), legend=False)
            else:
                fig = go.Figure()
                for clase, col in [("No viola", "no_violan"), ("Viola", "violan")]:
                    fig.add_bar(y=g["subreddit"], x=g[col], name=clase, orientation="h",
                                marker_color=E.CLASS_COLORS[clase],
                                hovertemplate="<b>%{x}</b> comentarios<br>r/%{y}<extra>" + clase + "</extra>")
                fig.update_layout(barmode="stack")
                fig.update_xaxes(title="Comentarios")
                mostrar(fig, "fig_subs", height=max(320, 26 * len(g) + 80))
            if "soccerstreams" in g["subreddit"].values:
                E.hallazgo("<b>r/soccerstreams</b> casi nunca viola la regla aunque está lleno de links: en esa "
                           "comunidad compartir streams es lo esperado. El subreddit (contexto) importa tanto "
                           "como el texto.")
            tabla(g.rename(columns={"n": "comentarios", "tasa": "tasa_violacion"}).round(3))

        st.write("")
        # ---- 3. señales binarias ----
        with st.container(key="card_flags"):
            E.seccion("Señales del texto: URL y palabras clave",
                      "Tasa de violación con y sin cada señal")
            a, b = st.columns([1.5, 1])
            flag = a.selectbox("Señal", BINARY_COLS, format_func=NOMBRES_VAR.get, index=2, key="ex_flag")
            separar = b.toggle("Separar por regla", value=True, key="ex_sep",
                               help="Controla el efecto de la regla (paradoja de Simpson)")
            grp = ["regla", flag] if separar else [flag]
            t = df.groupby(grp)[TARGET].agg(tasa="mean", n="size").reset_index()
            t["señal"] = np.where(t[flag] == 1, "Con la señal", "Sin la señal")
            fig = px.bar(t, x="señal", y=100 * t["tasa"], facet_col="regla" if separar else None,
                         text=t["tasa"].map(lambda v: pct(v)), custom_data=["n"],
                         category_orders={"señal": ["Sin la señal", "Con la señal"]},
                         color_discrete_sequence=[E.ORANGE])
            fig.update_traces(hovertemplate="<b>%{y:.1f}%</b> violan<br>%{x}<br>n = %{customdata[0]}<extra></extra>",
                              textposition="outside", cliponaxis=False)
            fig.update_yaxes(title="% que viola la regla", range=[0, 105])
            fig.update_xaxes(title="")
            fig.for_each_annotation(lambda an: an.update(text=an.text.split("=")[-1]))
            mostrar(fig, "fig_flags", legend=False)
            if flag == "has_legal_keyword":
                E.hallazgo("La palabra clave legal dispara la tasa de violación <b>solo bajo la regla legal</b>; "
                           "bajo No Advertising casi no aparece (n muy pequeño). Desactiva “Separar por regla” "
                           "para ver cómo la mezcla de reglas exagera el efecto.")
            elif flag == "has_url":
                E.hallazgo("Sin separar, la URL parece reducir las violaciones. Al separar por regla la "
                           "diferencia se achica: es un efecto de composición (casi todas las URL están en "
                           "No Advertising, la regla con menor tasa de violación).")
            else:
                E.hallazgo("Las palabras de spam (free, stream, download…) aparecen sobre todo en links de "
                           "streaming legítimos, por eso no predicen violación por sí solas.")
            tabla(t.drop(columns=[flag]).round(3))

        st.write("")
        # ---- 4. palabras ----
        with st.container(key="card_words"):
            E.seccion("¿Qué palabras usan?", "Términos lematizados (sin stopwords) por clase")
            a, b, c3 = st.columns([1.2, 1.2, 1.2])
            n_pal = a.slider("Cantidad de palabras", 8, 30, 15, key="ex_npal")
            criterio = b.segmented_control("Criterio", ["Más frecuentes", "Más distintivas"],
                                           default="Más distintivas", key="ex_crit") or "Más distintivas"
            medida = c3.segmented_control("Medida", ["% de comentarios", "Conteo"], default="% de comentarios",
                                          key="ex_medida") or "% de comentarios"
            cvz = CountVectorizer(binary=True, min_df=3)
            try:
                M = cvz.fit_transform(df["body_lemmatized"])
                vocab = cvz.get_feature_names_out()
                filas = []
                for clase in E.CLASS_ORDER:
                    mask = (df["clase"] == clase).values
                    if mask.sum():
                        cnt = np.asarray(M[mask].sum(axis=0)).ravel()
                        filas.append(pd.DataFrame({"palabra": vocab, "clase": clase, "conteo": cnt,
                                                   "pct": 100 * cnt / mask.sum()}))
                w = pd.concat(filas)
                piv = w.pivot(index="palabra", columns="clase", values="pct").fillna(0)
                if criterio == "Más distintivas" and piv.shape[1] == 2:
                    score = (piv["Viola"] - piv["No viola"]).abs()
                else:
                    score = piv.max(axis=1)
                top = score.sort_values(ascending=False).head(n_pal).index
                w = w[w["palabra"].isin(top)]
                orden_pal = score.loc[top].sort_values().index.tolist()
                ycol = "pct" if medida == "% de comentarios" else "conteo"
                fig = px.bar(w, y="palabra", x=ycol, color="clase", barmode="group", orientation="h",
                             color_discrete_map=E.CLASS_COLORS,
                             category_orders={"palabra": orden_pal, "clase": E.CLASS_ORDER},
                             labels={"pct": "% de comentarios de la clase que la usan", "conteo": "Comentarios",
                                     "palabra": "", "clase": "Clase"})
                fig.update_traces(hovertemplate="<b>%{x:.1f}</b><br>%{y}<extra>%{fullData.name}</extra>")
                mostrar(fig, "fig_words", height=max(360, 30 * n_pal + 80))
                E.hallazgo("Las violaciones se concentran en vocabulario legal (<i>lawyer, legal, sue, court</i>) y "
                           "promocional; los no violatorios incluyen jerga de streaming (<i>stream, hd, mobile</i>). "
                           "Prueba filtrar una sola regla arriba para ver cómo cambian.")
                tabla(w.pivot_table(index="palabra", columns="clase", values=["conteo", "pct"]).round(2).reset_index())
            except ValueError:
                st.info("No hay suficientes palabras en la selección actual.")

        st.write("")
        c_a, c_b = st.columns([1, 1.25], gap="medium")
        # ---- 5. correlaciones ----
        with c_a:
            with st.container(key="card_corr"):
                E.seccion("Correlaciones", "Variables derivadas vs la variable objetivo")
                metodo = st.segmented_control("Método", ["Pearson", "Spearman"], default="Pearson",
                                              key="ex_corr") or "Pearson"
                cols = NUMERIC_COLS + BINARY_COLS + [TARGET]
                cm = df[cols].corr(method=metodo.lower()).round(2)
                etiquetas = [NOMBRES_VAR[c] for c in cols]
                fig = go.Figure(go.Heatmap(z=cm.values, x=etiquetas, y=etiquetas, zmin=-1, zmax=1,
                                           colorscale=E.DIV_SCALE, text=cm.values, texttemplate="%{text:.2f}",
                                           textfont=dict(size=10),
                                           hovertemplate="<b>%{z:.2f}</b><br>%{y} × %{x}<extra></extra>",
                                           colorbar=dict(thickness=10, outlinewidth=0)))
                fig.update_yaxes(autorange="reversed")
                fig.update_xaxes(tickangle=40)
                mostrar(fig, "fig_corr", height=500, legend=False)
                tabla(cm.reset_index().rename(columns={"index": "variable"}))
        # ---- 6. explorador de comentarios ----
        with c_b:
            with st.container(key="card_tabla"):
                E.seccion("Explorador de comentarios", "El dataset limpio, tal como lo ven los modelos")
                cols_disp = ["body", "clase", "regla", "subreddit", "body_clean", "body_lemmatized"] + NUMERIC_COLS + BINARY_COLS
                cols_sel = st.multiselect("Columnas", cols_disp, default=["body", "clase", "regla", "subreddit",
                                                                         "char_length", "has_legal_keyword"],
                                          key="ex_cols")
                st.dataframe(df[cols_sel or ["body"]].head(500), hide_index=True, height=440,
                             column_config={"body": st.column_config.TextColumn("comentario", width="large"),
                                            "caps_ratio": st.column_config.NumberColumn(format="%.2f")})
                st.caption("Se muestran hasta 500 filas de la selección.")

        st.write("")
        with st.expander("Variables con las que se entrenaron los modelos"):
            st.dataframe(pd.DataFrame([
                ("text_rule", "Texto", "body_lemmatized + copia con prefijo de la regla (ad_/legal_)", "LR, NB, SVM, RF, XGBoost (TF-IDF)"),
                ("body", "Texto", "Comentario original (sin limpiar)", "SBERT (embeddings de 384 dimensiones)"),
                ("char_length, word_count, url_count, exclamation_count, emoji_count, caps_ratio", "Numéricas",
                 "Estadísticas del texto original", "Todos (estandarizadas)"),
                ("has_url, has_spam_keyword, has_legal_keyword", "Binarias", "Señales por palabra completa", "Todos"),
                ("rule_short", "Categórica", "Regla evaluada (one-hot)", "Todos"),
                ("subreddit", "Categórica", "Comunidad (one-hot, mínimo 3 apariciones)", "Todos"),
            ], columns=["Variable", "Tipo", "Descripción", "Modelos"]), hide_index=True)


# =====================================================================
# CLASIFICAR
# =====================================================================
def cargar_ejemplo():
    ej = st.session_state.get("cl_ejemplo")
    if ej in EJEMPLOS:
        sub, regla, texto = EJEMPLOS[ej]
        st.session_state["cl_sub"] = sub
        st.session_state["cl_regla"] = regla
        st.session_state["cl_texto"] = texto


def contribuciones_lr(X):
    """Aporte de cada variable a la predicción de la Regresión Logística (coef × valor)."""
    pipe = cargar_modelo("Regresión Logística")
    feats, clf = pipe.named_steps["feats"], pipe.named_steps["clf"]
    xt = feats.transform(X)
    xt = xt.toarray()[0] if hasattr(xt, "toarray") else np.asarray(xt)[0]
    contrib = pd.Series(xt * clf.coef_[0], index=feats.get_feature_names_out())
    contrib = contrib[contrib.abs() > 1e-6]

    def nombre(n):
        grupo, v = n.split("__", 1)
        if grupo == "tfidf":
            return f"“{v}”"
        if grupo == "num":
            return NOMBRES_VAR.get(v, v)
        if grupo == "rule":
            return f"regla: {v.replace('rule_short_', '')}"
        return f"r/{v.replace('subreddit_', '')}"
    contrib.index = [nombre(n) for n in contrib.index]
    return contrib.groupby(level=0).sum()


with tab_clasificar:
    st.session_state.setdefault("historial", [])
    st.session_state.setdefault("cl_regla", "No Advertising")
    st.session_state.setdefault("cl_sub", "AskReddit")
    st.session_state.setdefault("cl_texto", "")
    izq, der = st.columns([1, 1.15], gap="large")
    with izq:
        with st.container(key="card_post"):
            E.seccion("Crea un post", "El preprocesamiento se hace automáticamente")
            st.pills("Cargar un ejemplo real del conjunto de prueba", list(EJEMPLOS), key="cl_ejemplo",
                     on_change=cargar_ejemplo)
            subs_lista = train["subreddit"].value_counts().index.tolist()
            if st.session_state["cl_sub"] not in subs_lista:
                subs_lista = [st.session_state["cl_sub"]] + subs_lista
            a, b = st.columns([1, 1])
            sub = a.selectbox("Subreddit", subs_lista, key="cl_sub", format_func=lambda s: f"r/{s}")
            otro = b.text_input("¿Otro subreddit?", key="cl_sub_otro", placeholder="opcional, ej. gaming")
            subreddit = otro.strip().removeprefix("r/") or sub
            regla = st.segmented_control("Regla que se evalúa", list(REGLAS), key="cl_regla") or "No Advertising"
            st.caption(f"*{REGLAS[regla]}*")
            texto = st.text_area("Comentario (en inglés, como en el dataset)", key="cl_texto", height=140,
                                 placeholder="Write a Reddit comment…")
            usar_todos = st.toggle("Usar todos los modelos", value=True, key="cl_todos")
            if usar_todos:
                elegidos = MODELOS
            else:
                elegidos = st.pills("Modelos", MODELOS, selection_mode="multi", default=[mejor],
                                    key="cl_modelos") or []
            with st.expander("Opciones avanzadas"):
                umbral = st.slider("Umbral de decisión", 0.05, 0.95, 0.50, 0.05, key="cl_umbral",
                                   help="Probabilidad a partir de la cual se considera que el comentario viola la regla")
            analizar = st.button("Analizar comentario", type="primary", width="stretch",
                                 disabled=not texto.strip() or not elegidos)

    if analizar:
        X = preprocess_input(texto, REGLAS[regla], subreddit)
        filas, errores = [], []
        for m in elegidos:
            try:
                pipe = cargar_modelo(m)
                t0 = time.perf_counter()
                p = float(pipe.predict_proba(X)[:, 1][0])
                filas.append({"modelo": m, "probabilidad": p, "ms": 1000 * (time.perf_counter() - t0)})
            except Exception as e:   # p. ej. SBERT sin internet ni copia local
                errores.append(f"{m}: {e}")
        st.session_state["resultado"] = {"X": X, "filas": filas, "errores": errores, "texto": texto,
                                         "regla": regla, "subreddit": subreddit}
        for f in filas:
            st.session_state["historial"].insert(0, {"comentario": texto[:80], "regla": regla,
                                                     "subreddit": subreddit, "modelo": f["modelo"],
                                                     "probabilidad": round(f["probabilidad"], 3)})

    with der:
        res = st.session_state.get("resultado")
        if not res:
            with st.container(key="card_vacio"):
                E.seccion("Resultado")
                st.markdown("Escribe un comentario o carga un ejemplo y presiona **Analizar comentario**. "
                            "Verás la probabilidad que asigna cada modelo y en qué coinciden.")
                st.caption("Tip: carga “Link de streaming” y luego “Spam de regalos”: ambos tienen links, "
                           "pero solo uno es spam.")
        else:
            for err in res["errores"]:
                st.warning(f"No se pudo usar el modelo — {err}")
            r = pd.DataFrame(res["filas"])
            if len(r):
                r["veredicto"] = np.where(r["probabilidad"] >= umbral, "Viola", "No viola")
                r = r.merge(met[["modelo", "roc_auc", "cv_roc_auc"]], on="modelo", how="left")
                n_v = int((r["veredicto"] == "Viola").sum())
                p_media = r["probabilidad"].mean()
                viola = p_media >= umbral
                st.markdown(f"""
<div class="rd-verdict {'viola' if viola else 'noviola'}">
  <div class="small">r/{E.esc(res['subreddit'])} · {E.esc(res['regla'])}</div>
  <div class="big"><span class="rd-dot" style="background:{E.ORANGE if viola else E.PERI}"></span>{'Probablemente VIOLA la regla' if viola else 'Probablemente NO viola la regla'}</div>
  <div class="small"><b>{n_v} de {len(r)}</b> modelos dicen que viola · probabilidad promedio <b>{pct(p_media)}</b> · umbral {pct(umbral, 0)}</div>
</div>""", unsafe_allow_html=True)
                st.write("")
                with st.container(key="card_resultado"):
                    E.seccion("Probabilidad de violación por modelo")
                    r_plot = r.sort_values("probabilidad")
                    fig = go.Figure(go.Bar(
                        y=r_plot["modelo"], x=100 * r_plot["probabilidad"], orientation="h",
                        marker_color=[E.model_color(m) for m in r_plot["modelo"]],
                        text=[f"{100 * p:.0f}% · {v}" for p, v in zip(r_plot["probabilidad"], r_plot["veredicto"])],
                        textposition="outside", cliponaxis=False,
                        customdata=r_plot[["roc_auc", "ms"]],
                        hovertemplate="<b>%{x:.1f}%</b><br>%{y}<br>AUC holdout %{customdata[0]:.3f} · "
                                      "%{customdata[1]:.1f} ms<extra></extra>"))
                    fig.add_vline(x=100 * umbral, line_dash="dot", line_color=E.MUTED)
                    fig.update_xaxes(range=[0, 118],
                                     title=f"Probabilidad de violar la regla (%) · línea punteada = umbral {pct(umbral, 0)}")
                    mostrar(fig, "fig_pred", height=max(240, 48 * len(r) + 90), legend=False)
                    st.dataframe(r[["modelo", "probabilidad", "veredicto", "roc_auc", "ms"]], hide_index=True,
                                 column_config={
                                     "probabilidad": st.column_config.ProgressColumn("probabilidad", min_value=0, max_value=1, format="percent"),
                                     "veredicto": st.column_config.TextColumn("veredicto", width="small"),
                                     "roc_auc": st.column_config.NumberColumn("AUC", format="%.3f", width="small"),
                                     "ms": st.column_config.NumberColumn("ms", format="%.1f", width="small")})
                with st.expander("Cómo ve el modelo tu comentario (preprocesamiento)"):
                    X = res["X"]
                    st.markdown(f"**Texto limpio:** {E.esc(X['body_clean'].iloc[0]) or '—'}")
                    st.markdown(f"**Lematizado:** {E.esc(X['body_lemmatized'].iloc[0]) or '—'}")
                    st.dataframe(X[NUMERIC_COLS + BINARY_COLS].rename(columns=NOMBRES_VAR), hide_index=True)
                if "Regresión Logística" in MODELOS:
                    with st.expander("¿Por qué? Lo que más pesó en la Regresión Logística"):
                        try:
                            ctb = contribuciones_lr(res["X"])
                            top = ctb.reindex(ctb.abs().sort_values(ascending=False).head(12).index).sort_values()
                            fig = go.Figure(go.Bar(
                                y=top.index, x=top.values, orientation="h",
                                marker_color=[E.ORANGE if v > 0 else E.PERI for v in top.values],
                                hovertemplate="%{y}<br>aporte <b>%{x:+.3f}</b><extra></extra>"))
                            fig.add_vline(x=0, line_color=E.LINE)
                            fig.update_xaxes(title="← empuja a “no viola”   ·   empuja a “viola” →")
                            mostrar(fig, "fig_contrib", height=max(260, 28 * len(top) + 80), legend=False)
                            st.caption("Aporte = coeficiente × valor de la variable. Las palabras con prefijo "
                                       "(ad_/legal_) son específicas de la regla evaluada.")
                        except Exception as e:
                            st.info(f"No disponible: {e}")

    if st.session_state["historial"]:
        st.write("")
        with st.expander(f"Historial de esta sesión ({len(st.session_state['historial'])} predicciones)"):
            st.dataframe(pd.DataFrame(st.session_state["historial"]), hide_index=True,
                         column_config={"probabilidad": st.column_config.ProgressColumn(
                             "probabilidad", min_value=0, max_value=1, format="percent")})
            if st.button("Borrar historial", key="cl_borrar"):
                st.session_state["historial"] = []
                st.rerun()


# =====================================================================
# RENDIMIENTO
# =====================================================================
with tab_rend:
    mostrar_rend = st.toggle("Mostrar el rendimiento de los modelos", value=True, key="rd_mostrar")
    if not mostrar_rend:
        st.info("La información de rendimiento está oculta. Activa el interruptor para verla.")
    else:
        y_true = hold["y_true"].values
        with st.container(key="filtros_rend"):
            a, b = st.columns([3, 1.2])
            sel_mod = a.pills("Modelos", MODELOS, selection_mode="multi", default=MODELOS, key="rd_modelos")
            umbral_r = b.slider("Umbral de decisión", 0.05, 0.95, 0.50, 0.05, key="rd_umbral")
        sel_mod = [m for m in MODELOS if m in (sel_mod or [])]
        st.caption(f"Evaluación sobre el **holdout** ({len(hold)} comentarios, partición agrupada por texto). "
                   "Los filtros aplican a todas las gráficas. Haz clic en la leyenda para ocultar series.")
        if not sel_mod:
            st.warning("Selecciona al menos un modelo.")
        else:
            def metricas_umbral(m, u):
                p = hold[f"proba_{m}"].values
                pred = (p >= u).astype(int)
                return {"modelo": m, "accuracy": accuracy_score(y_true, pred),
                        "precision": precision_score(y_true, pred, zero_division=0),
                        "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred),
                        "roc_auc": roc_auc_score(y_true, p)}

            mu = pd.DataFrame([metricas_umbral(m, umbral_r) for m in sel_mod]).merge(
                met[["modelo", "cv_roc_auc", "tiempo_entrenamiento_s", "tiempo_prediccion_ms",
                     "tiempo_tuneo_s", "mejores_parametros"]], on="modelo")

            k = st.columns(3)
            b_cv = mu.loc[mu["cv_roc_auc"].idxmax()]
            b_ho = mu.loc[mu["roc_auc"].idxmax()]
            b_t = mu.loc[mu["tiempo_prediccion_ms"].idxmin()]
            with k[0]:
                E.stat("Mejor AUC en validación cruzada", f"{b_cv['cv_roc_auc']:.3f}", b_cv["modelo"])
            with k[1]:
                E.stat("Mejor AUC en el holdout", f"{b_ho['roc_auc']:.3f}", b_ho["modelo"])
            with k[2]:
                E.stat("Predicción más rápida", f"{b_t['tiempo_prediccion_ms']:.2f} ms", b_t["modelo"])
            st.write("")
            E.hallazgo("Las diferencias de AUC entre los mejores modelos (≈0.01) son menores que la desviación "
                       "entre particiones (±0.02, ver variantes): <b>SBERT</b> gana por validación cruzada, pero la "
                       "<b>Regresión Logística</b> es casi igual de buena, mucho más rápida e interpretable.")

            # ---- métricas ----
            with st.container(key="card_metricas"):
                E.seccion("Comparación de métricas", f"Umbral {umbral_r:.2f} (AUC no depende del umbral)")
                a, b = st.columns([3, 1])
                mets = a.multiselect("Métricas", ["accuracy", "precision", "recall", "f1", "roc_auc", "cv_roc_auc"],
                                     default=["accuracy", "precision", "recall", "f1", "roc_auc"], key="rd_mets")
                forma = b.segmented_control("Forma", ["Barras", "Puntos"], default="Barras", key="rd_forma") or "Barras"
                long = mu.melt(id_vars="modelo", value_vars=mets or ["roc_auc"], var_name="métrica", value_name="valor")
                if forma == "Barras":
                    fig = px.bar(long, x="métrica", y="valor", color="modelo", barmode="group",
                                 color_discrete_map=E.MODEL_COLORS, category_orders={"modelo": sel_mod})
                    fig.update_yaxes(range=[0, 1], title="Valor")
                else:   # puntos: el eje puede acercarse sin exagerar diferencias
                    fig = px.strip(long, x="métrica", y="valor", color="modelo",
                                   color_discrete_map=E.MODEL_COLORS, category_orders={"modelo": sel_mod})
                    fig.update_traces(marker=dict(size=13, line=dict(width=2, color=E.SURFACE)), jitter=0.35)
                    fig.update_yaxes(title="Valor (eje acercado)")
                fig.update_traces(hovertemplate="<b>%{y:.3f}</b><br>%{x}<extra>%{fullData.name}</extra>")
                fig.update_xaxes(title="")
                mostrar(fig, "fig_mets", height=400)
                tabla(mu.drop(columns=["mejores_parametros"]).round(4), label="Ver tabla de métricas")

            st.write("")
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                with st.container(key="card_roc"):
                    E.seccion("Curvas ROC", "Cuanto más arriba a la izquierda, mejor")
                    fig = go.Figure()
                    roc_rows = []
                    for m in sel_mod:
                        fpr, tpr, thr = roc_curve(y_true, hold[f"proba_{m}"])
                        auc = roc_auc_score(y_true, hold[f"proba_{m}"])
                        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{m} ({auc:.3f})",
                                        line=dict(color=E.model_color(m), width=2),
                                        customdata=np.clip(thr, 0, 1),
                                        hovertemplate="TPR <b>%{y:.2f}</b> · FPR %{x:.2f}<br>umbral %{customdata:.2f}"
                                                      f"<extra>{m}</extra>")
                        roc_rows.append({"modelo": m, "AUC": round(auc, 4)})
                    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                                    line=dict(color=E.LINE, dash="dash", width=1), hoverinfo="skip")
                    fig.update_xaxes(title="Tasa de falsos positivos", range=[0, 1])
                    fig.update_yaxes(title="Tasa de verdaderos positivos", range=[0, 1.02])
                    fig.update_layout(legend=dict(orientation="v", y=0.02, x=0.98, xanchor="right", yanchor="bottom",
                                                  bgcolor=E.LEGEND_BG))
                    mostrar(fig, "fig_roc", height=430)
                    tabla(pd.DataFrame(roc_rows))
            with c2:
                with st.container(key="card_cm"):
                    E.seccion("Matriz de confusión", "Aciertos y errores con el umbral elegido")
                    a, b = st.columns([1.4, 1])
                    m_cm = a.selectbox("Modelo", sel_mod, key="rd_cm_modelo")
                    unid = b.segmented_control("Unidades", ["Conteos", "% por clase real"], default="Conteos",
                                               key="rd_cm_unid") or "Conteos"
                    cmx = confusion_matrix(y_true, (hold[f"proba_{m_cm}"] >= umbral_r).astype(int))
                    z = cmx / cmx.sum(axis=1, keepdims=True) * 100 if unid != "Conteos" else cmx
                    txt = [[f"{v:.1f}%" if unid != "Conteos" else f"{v:d}" for v in fila] for fila in z]
                    fig = go.Figure(go.Heatmap(
                        z=z, x=["Predice: No viola", "Predice: Viola"], y=["Real: No viola", "Real: Viola"],
                        colorscale=E.SEQ_SCALE, text=txt, texttemplate="%{text}", textfont=dict(size=18),
                        showscale=False, xgap=3, ygap=3,
                        hovertemplate="%{y} · %{x}<br><b>%{text}</b><extra></extra>"))
                    fig.update_yaxes(autorange="reversed")
                    mostrar(fig, "fig_cm", height=340, legend=False)
                    tn, fp, fn, tp = cmx.ravel()
                    E.hallazgo(f"Con umbral {umbral_r:.2f}, {E.esc(m_cm)} deja pasar <b>{fn}</b> violaciones "
                               f"(falsos negativos) y marca <b>{fp}</b> comentarios inocentes (falsos positivos). "
                               "Sube el umbral para molestar menos a los usuarios; bájalo para no dejar pasar violaciones.")

            st.write("")
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                with st.container(key="card_umbral"):
                    E.seccion("¿Qué pasa al mover el umbral?")
                    m_u = st.selectbox("Modelo", sel_mod, key="rd_u_modelo")
                    us = np.round(np.arange(0.05, 0.96, 0.01), 2)
                    cu = pd.DataFrame([metricas_umbral(m_u, u) | {"umbral": u} for u in us])
                    fig = go.Figure()
                    estilos = {"f1": ("F1", E.model_color(m_u), "solid", 3), "precision": ("Precisión", E.INK2, "dash", 2),
                               "recall": ("Recall", E.MUTED, "dot", 2), "accuracy": ("Exactitud", E.ERR, "dashdot", 2)}
                    for col, (nom, color, dash, w) in estilos.items():
                        fig.add_scatter(x=cu["umbral"], y=cu[col], name=nom, mode="lines",
                                        line=dict(color=color, dash=dash, width=w),
                                        hovertemplate=f"{nom} <b>%{{y:.3f}}</b><extra></extra>")
                    fig.add_vline(x=umbral_r, line_color=E.ORANGE, line_width=1)
                    fig.update_layout(hovermode="x unified")
                    fig.update_xaxes(title="Umbral de decisión")
                    fig.update_yaxes(title="Valor", range=[0, 1.02])
                    mostrar(fig, "fig_umbral", height=380)
                    tabla(cu.drop(columns=["modelo", "roc_auc"]).round(3))
            with c2:
                with st.container(key="card_probs"):
                    E.seccion("Distribución de probabilidades", "¿Separa bien el modelo las dos clases?")
                    a, b = st.columns([1.4, 1])
                    m_p = a.selectbox("Modelo", sel_mod, key="rd_p_modelo")
                    unid_p = b.segmented_control("Unidades", ["Conteo", "Porcentaje"], default="Conteo",
                                                 key="rd_p_unid") or "Conteo"
                    dp = pd.DataFrame({"probabilidad": hold[f"proba_{m_p}"],
                                       "clase real": np.where(y_true == 1, "Viola", "No viola")})
                    fig = px.histogram(dp, x="probabilidad", color="clase real", nbins=20, barmode="group",
                                       histnorm="percent" if unid_p == "Porcentaje" else None,
                                       color_discrete_map=E.CLASS_COLORS,
                                       category_orders={"clase real": E.CLASS_ORDER})
                    fig.add_vline(x=umbral_r, line_color=E.INK2, line_dash="dot")
                    fig.update_yaxes(title="% de cada clase" if unid_p == "Porcentaje" else "Comentarios")
                    fig.update_xaxes(title="Probabilidad predicha de violar la regla")
                    mostrar(fig, "fig_probs", height=380)

            st.write("")
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                with st.container(key="card_regla"):
                    E.seccion("AUC por regla", "¿Los modelos rinden igual en ambas reglas?")
                    ar = []
                    for m in sel_mod:
                        for rg in ["No Advertising", "No legal advice"]:
                            mask = (hold["regla"] == rg).values
                            ar.append({"modelo": m, "regla": rg,
                                       "AUC": roc_auc_score(y_true[mask], hold.loc[mask, f"proba_{m}"])})
                    ar = pd.DataFrame(ar)
                    fig = px.bar(ar, x="regla", y="AUC", color="modelo", barmode="group",
                                 color_discrete_map=E.MODEL_COLORS, category_orders={"modelo": sel_mod})
                    fig.update_traces(hovertemplate="<b>%{y:.3f}</b><br>%{x}<extra>%{fullData.name}</extra>")
                    fig.update_yaxes(range=[0, 1], title="ROC-AUC")
                    fig.update_xaxes(title="")
                    mostrar(fig, "fig_regla", height=380)
                    tabla(ar.round(4))
            with c2:
                with st.container(key="card_tiempo"):
                    E.seccion("Efectividad vs tiempo")
                    opciones_t = {"Entrenamiento (s)": "tiempo_entrenamiento_s",
                                  "Predicción por comentario (ms)": "tiempo_prediccion_ms",
                                  "Búsqueda de hiperparámetros (s)": "tiempo_tuneo_s"}
                    tsel = st.segmented_control("Tiempo", list(opciones_t), default="Predicción por comentario (ms)",
                                                key="rd_tiempo") or "Predicción por comentario (ms)"
                    col_t = opciones_t[tsel]
                    # un solo color + etiquetas directas: en un scatter cualquier par de puntos puede quedar cerca
                    fig = go.Figure(go.Scatter(
                        x=mu[col_t], y=mu["roc_auc"], mode="markers+text", text=[E.MODEL_SHORT.get(m, m) for m in mu["modelo"]],
                        textposition="top center", marker=dict(size=14, color=E.NEUTRAL_MARK, line=dict(width=2, color=E.SURFACE)),
                        customdata=mu["modelo"],
                        hovertemplate="<b>%{customdata}</b><br>AUC %{y:.3f}<br>" + tsel + ": %{x:.3f}<extra></extra>"))
                    fig.update_xaxes(type="log", title=tsel + " · escala log")
                    fig.update_yaxes(title="ROC-AUC (holdout)")
                    mostrar(fig, "fig_tiempo", height=380, legend=False)
                    tabla(mu[["modelo", "roc_auc", col_t]].round(4))

            if variantes is not None:
                st.write("")
                with st.container(key="card_variantes"):
                    E.seccion("¿Importa cómo se limpia el texto?",
                              "Mismo split y misma búsqueda; media ± desviación de 5×5 particiones agrupadas")
                    filtro_v = st.segmented_control("Variantes", ["Todas", "Con variables extra", "Solo texto"],
                                                    default="Todas", key="rd_var") or "Todas"
                    v = variantes.copy()
                    if filtro_v == "Con variables extra":
                        v = v[v["features_extra"]]
                    elif filtro_v == "Solo texto":
                        v = v[~v["features_extra"]]
                    v["etiqueta"] = [f"{t}{' + prefijo' if p else ''}{' + extras' if x else ' (solo texto)'}"
                                     for t, p, x in zip(v["texto"], v["prefijo_regla"], v["features_extra"])]
                    v = v.sort_values("auc_rep_media")
                    fig = go.Figure()
                    for m in v["modelo"].unique():
                        s = v[v["modelo"] == m]
                        fig.add_scatter(x=s["auc_rep_media"], y=s["etiqueta"], mode="markers", name=m,
                                        marker=dict(size=12, color=E.model_color(m), line=dict(width=2, color=E.SURFACE)),
                                        error_x=dict(type="data", array=s["auc_rep_std"], color=E.ERR, thickness=1.5),
                                        customdata=s[["auc_rep_std", "holdout_auc"]],
                                        hovertemplate="<b>%{x:.4f}</b> ± %{customdata[0]:.4f}<br>holdout %{customdata[1]:.4f}"
                                                      "<extra>%{y}</extra>")
                    fig.update_yaxes(categoryorder="array", categoryarray=v["etiqueta"].tolist(), title="")
                    fig.update_xaxes(title="ROC-AUC (media ± desviación)")
                    mostrar(fig, "fig_var", height=max(320, 30 * len(v) + 90))
                    E.hallazgo("Agregar la regla y el subreddit aporta mucho más que la forma de limpiar el texto. "
                               "Con variables extra, texto sin limpiar, limpio y lematizado quedan dentro de una "
                               "desviación estándar entre sí; el prefijo de regla ayuda de forma consistente.")
                    tabla(v.drop(columns=["etiqueta"]).round(4))

            if cv_tablas:
                st.write("")
                with st.container(key="card_tuneo"):
                    E.seccion("Ajuste de hiperparámetros", "GridSearchCV con 5 folds agrupados por comentario")
                    m_t = st.selectbox("Modelo", [m for m in sel_mod if m in cv_tablas], key="rd_tuneo")
                    if m_t:
                        ct = cv_tablas[m_t].copy()
                        ct["combinación"] = ct["params"].str.replace(r"clf__|feats__tfidf__", "", regex=True)
                        ct = ct.sort_values("mean_test_score")
                        fig = go.Figure(go.Scatter(
                            x=ct["mean_test_score"], y=ct["combinación"], mode="markers",
                            marker=dict(size=11, color=E.model_color(m_t), line=dict(width=2, color=E.SURFACE)),
                            error_x=dict(type="data", array=ct["std_test_score"], color=E.ERR, thickness=1.5),
                            hovertemplate="AUC <b>%{x:.4f}</b><br>%{y}<extra></extra>"))
                        fig.update_xaxes(title="AUC promedio en validación cruzada (± desviación)")
                        fig.update_yaxes(title="", tickfont=dict(size=11))
                        mostrar(fig, "fig_tuneo", height=max(300, 26 * len(ct) + 90), legend=False)
                        st.caption(f"Mejores parámetros: `{met.set_index('modelo').loc[m_t, 'mejores_parametros']}`")
                        tabla(ct[["combinación", "mean_test_score", "std_test_score", "rank_test_score",
                                  "mean_fit_time"]].round(4))
