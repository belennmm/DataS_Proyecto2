# Proyecto 2 – Clasificación de reglas de comunidad (Jigsaw)


## Reto seleccionado
**#4 – Jigsaw: Agile Community Rules Classification**
Tema: Procesamiento del Lenguaje Natural

Dado el texto de un comentario de Reddit, la regla de moderación de un subreddit y un par de ejemplos de referencia (positivos y negativos), el objetivo es predecir la probabilidad de que ese comentario viole la regla indicada.

## Integrantes del grupo y contribuidores al repositorio
- André pivaral  
- Belén Monterroso 
- Melisa Mendizabal 
- Renato Rojas 

## Estructura del repositorio
```
.
├── Datos/
│   ├── train.csv             # dataset original de entrenamiento (2029 filas, 9 columnas)
│   ├── test.csv              # dataset original de prueba (10 filas, sin la columna objetivo)
│   ├── train_clean.csv       # train con limpieza y variables derivadas ya aplicadas
│   └── test_clean.csv        # test con la misma limpieza y variables derivadas
├── figuras/                  # gráficos del EDA (01-10) y de los modelos (modelos_01-07)
├── modelos/                  # pipelines entrenados (.joblib) + modelo_final.joblib/.json
├── resultados/               # métricas, tuneo (cv_*.csv), variantes, predicciones y submission
├── Proyecto2Analisis.ipynb   # notebook con limpieza, preprocesamiento y EDA completo
├── preprocesamiento.py       # limpieza y variables derivadas (misma lógica del notebook), usada por modelos y app
├── variantes.py              # experimento: texto sin limpiar vs limpio vs lematizado
├── modelos.py                # entrenamiento, tuneo, evaluación y exportación de los modelos
├── app.py                    # aplicación interactiva (Streamlit + Plotly)
├── app_estilo.py             # estética de la app (CSS tipo Reddit, plantilla de Plotly)
├── paleta.py                 # paleta de colores compartida por la app y las figuras
├── .streamlit/config.toml    # tema de la app
├── requirements.txt
└── README.md
```

## Datos
| Variable | Tipo | Descripción |
|---|---|---|
| row_id | numérica discreta | identificador único, no predictiva |
| body | texto | comentario de Reddit a clasificar |
| rule | categórica | regla de moderación evaluada |
| subreddit | categórica | comunidad de origen |
| positive_example_1/2 | texto | ejemplos que sí violan la regla, dados como contexto |
| negative_example_1/2 | texto | ejemplos que no violan la regla, dados como contexto |
| rule_violation | categórica binaria (**target**) | 0 = no viola, 1 = viola |

`train.csv` tiene 2029 observaciones y 9 variables. `test.csv` tiene 10 observaciones y 8 variables (no incluye `rule_violation`, que es lo que hay que predecir; `sample_submission.csv` pide una probabilidad por `row_id`, no una etiqueta binaria).

## Limpieza y preprocesamiento
Todo el proceso está documentado y justificado paso a paso en `Proyecto2Analisis.ipynb`. En resumen:
- **Nulos:** no se encontraron en ninguno de los dos datasets, por lo que no se aplicó imputación.
- **Duplicados:** se verificó que no hay duplicados exactos de fila completa (0 en ambos datasets). Se identificaron 57 comentarios (`body`) repetidos (217 filas); solo 6 de ellos se evaluaron con más de una regla, el resto se repite bajo la misma regla en otro subreddit o con otros ejemplos. **No se eliminaron**, ya que la unidad de análisis es el par (comentario, regla); para evitar fuga de información, la partición train/holdout y la validación cruzada agrupan por `body`.
- **Palabras clave:** se buscan por palabra completa (no por subcadena), para que por ejemplo "sue" no coincida con "issue".
- **Normalización de texto:** se generó `body_clean` (minúsculas, sin URLs, sin markdown de Reddit, sin emojis ni caracteres no alfabéticos) y `body_lemmatized` (tokenizado, sin stopwords en inglés y lematizado), este último pensado para alimentar directamente un vectorizador TF-IDF.
- **Variables derivadas** (calculadas sobre el texto original antes de limpiarlo, para no perder señal): `char_length`, `word_count`, `url_count`, `has_url`, `exclamation_count`, `emoji_count`, `caps_ratio`, `has_spam_keyword`, `has_legal_keyword`.
- **Idioma:** se verificó con detección automática que el corpus es mayoritariamente inglés.
- El mismo pipeline se aplicó de forma idéntica a `train` y a `test` para evitar inconsistencias entre entrenamiento y predicción.

## Hallazgos principales del EDA
- Dataset balanceado: 50.8% de los comentarios en train violan la regla evaluada.
- `has_legal_keyword` es la señal más fuerte encontrada: la tasa de violación sube de 42.3% a 87.5% cuando el comentario contiene palabras asociadas a asesoría legal (bajo la regla legal: 40.8% → 87.9%; bajo No Advertising casi no aparece).
- La presencia de URL o de palabras típicas de spam se asocia con una tasa de violación **menor** (43.0% vs 56.7%). Es sobre todo un efecto de composición: casi todas las URL se evalúan con "No Advertising" (la regla con menor tasa) y r/soccerstreams (96% con URL) casi nunca viola la regla; dentro de "No Advertising" la diferencia es pequeña (42.8% vs 45.8%).
- Hay diferencias marcadas por regla (43.3% de violación en "No Advertising" vs. 58.3% en "No legal advice") y por subreddit (desde 2.9% en soccerstreams hasta 81.0% en sex y 78.9% en legaladvice).
- Los hallazgos corresponden a la muestra específica del reto Jigsaw y no deben generalizarse a Reddit en su totalidad.

Detalle completo de estadísticas descriptivas, tablas de frecuencia, cruces de variables y gráficos en `Proyecto2Analisis.ipynb` y en la carpeta `figuras/`.

## Modelado
- **Modelos:** Regresión Logística, Naive Bayes, SVM, Random Forest, XGBoost (TF-IDF + variables derivadas + regla + subreddit) y SBERT (`all-MiniLM-L6-v2`) + Regresión Logística.
- **Texto:** `body_lemmatized` con prefijo de regla (`ad_`/`legal_`) para que una palabra pese distinto según la regla; SBERT usa el texto original.
- **Partición:** 80/20 con `StratifiedGroupKFold` agrupando por `body`; tuneo con `GridSearchCV` (5 folds agrupados, métrica ROC-AUC). El modelo final se elige por AUC de validación cruzada.
- **Ejemplos positivos/negativos:** no se usan como variables, porque los mismos textos aparecen como `body` en train (con la misma etiqueta en el 97% de los casos), lo que produciría fuga de etiqueta.
- **test.csv:** sus 10 comentarios también están en train, por lo que `submission.csv` solo sirve como formato; la evaluación se hace sobre el holdout.

## Cómo ejecutar el proyecto
1. Clonar el repositorio.
2. Colocar `train.csv` y `test.csv` dentro de la carpeta `Datos/` (o ajustar la ruta en el notebook/script si se usa Kaggle o Google Colab).
3. Instalar dependencias (para GPU, ver la nota de `torch` en `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```
4. Correr `Proyecto2Analisis.ipynb` de principio a fin → `Datos/*_clean.csv` y figuras del EDA.
5. `python variantes.py --sbert` → `resultados/variantes.csv` y `figuras/modelos_07_variantes.png`.
6. `python modelos.py --sbert` → modelos en `modelos/`, métricas en `resultados/` y figuras `modelos_01-06`.

XGBoost y SBERT usan la GPU automáticamente si hay CUDA disponible; los demás modelos corren en CPU. Los modelos se exportan una sola vez al final del entrenamiento (sin checkpoints intermedios).

7. `streamlit run app.py` → abre la aplicación en el navegador (http://localhost:8501).

## Aplicación
`app.py` (Streamlit + Plotly) tiene cuatro pestañas:
- **Inicio:** los hallazgos principales del EDA y del modelado, contados como posts de Reddit.
- **Explorar datos:** filtros por regla, clase, subreddit y texto que aplican a todas las gráficas; distribución de variables (histograma, caja o violín, en conteo o porcentaje), tasa de violación por subreddit, señales binarias controlando por regla, palabras más frecuentes o distintivas, correlaciones y un explorador del dataset limpio.
- **Clasificar comentario:** se escribe un comentario (o se carga un ejemplo real del holdout), se elige el subreddit, la regla y uno o todos los modelos. El preprocesamiento es invisible para el usuario (`preprocess_input`). Se muestra la probabilidad de cada modelo, el consenso, cómo se ve el texto preprocesado y qué palabras pesaron más en la Regresión Logística.
- **Rendimiento de modelos:** se puede ocultar con un interruptor. Métricas, curvas ROC, matriz de confusión (conteos o %), efecto del umbral, distribución de probabilidades, AUC por regla, tiempo vs AUC, variantes de preprocesamiento y resultados del ajuste de hiperparámetros.

Todas las gráficas son interactivas (hover, zoom, clic en la leyenda para ocultar series) y tienen una tabla con sus datos. Un botón en la barra superior alterna entre modo claro y modo oscuro. La paleta imita a Reddit con colores poco saturados: naranja = viola, azul periwinkle = no viola (par cálido/frío distinguible con daltonismo) y un color fijo por modelo, validado para daltonismo y contraste (`paleta.py`).

## Referencias
- [Jigsaw - Agile Community Rules Classification (Kaggle)](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)
- Guía del Proyecto 2 – CC3084 Data Science, UVG, Semestre II 2026.