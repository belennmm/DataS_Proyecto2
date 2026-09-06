# Proyecto 2 – Análisis Exploratorio


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
│   ├── train.csv            # dataset original de entrenamiento (2029 filas, 9 columnas)
│   ├── test.csv              # dataset original de prueba (10 filas, sin la columna objetivo)
│   ├── train_clean.csv       # train con limpieza y variables derivadas ya aplicadas
│   └── test_clean.csv        # test con la misma limpieza y variables derivadas
├── figuras/                  # gráficos exploratorios generados por el notebook (.png)
├── Proyecyo2Analisis.ipynb   # notebook con limpieza, preprocesamiento y EDA completo
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
Todo el proceso está documentado y justificado paso a paso en `Proyecyo2Analisis.ipynb`. En resumen:
- **Nulos:** no se encontraron en ninguno de los dos datasets, por lo que no se aplicó imputación.
- **Duplicados:** se eliminaron duplicados exactos de fila completa (0 en ambos datasets). Se identificaron 57 comentarios (`body`) que se repiten bajo distinta regla/subreddit; estos **no se eliminaron**, ya que la unidad de análisis del problema es el par (comentario, regla), no el comentario aislado.
- **Normalización de texto:** se generó `body_clean` (minúsculas, sin URLs, sin markdown de Reddit, sin emojis ni caracteres no alfabéticos) y `body_lemmatized` (tokenizado, sin stopwords en inglés y lematizado), este último pensado para alimentar directamente un vectorizador TF-IDF.
- **Variables derivadas** (calculadas sobre el texto original antes de limpiarlo, para no perder señal): `char_length`, `word_count`, `url_count`, `has_url`, `exclamation_count`, `emoji_count`, `caps_ratio`, `has_spam_keyword`, `has_legal_keyword`.
- **Idioma:** se verificó con detección automática que el corpus es mayoritariamente inglés.
- El mismo pipeline se aplicó de forma idéntica a `train` y a `test` para evitar inconsistencias entre entrenamiento y predicción.

## Hallazgos principales del EDA
- Dataset balanceado: 50.8% de los comentarios en train violan la regla evaluada.
- `has_legal_keyword` es la señal más fuerte encontrada: la tasa de violación sube de 42.8% a 85.9% cuando el comentario contiene palabras asociadas a asesoría legal.
- La presencia de URL o de palabras típicas de spam se asocia, contraintuitivamente, con una tasa de violación **menor**, lo que indica que muchos comentarios "no violatorios" también incluyen enlaces legítimos.
- Hay diferencias marcadas por regla (43.3% de violación en "No Advertising" vs. 58.3% en "No legal advice") y por subreddit (desde 2.9% en soccerstreams hasta 81.0% en sex/legaladvice).
- Los hallazgos corresponden a la muestra específica del reto Jigsaw y no deben generalizarse a Reddit en su totalidad.

Detalle completo de estadísticas descriptivas, tablas de frecuencia, cruces de variables y gráficos en `Proyecyo2Analisis.ipynb` y en la carpeta `figuras/`.

## Cómo ejecutar el proyecto
1. Clonar el repositorio.
2. Colocar `train.csv` y `test.csv` dentro de la carpeta `Datos/` (o ajustar la ruta en el notebook/script si se usa Kaggle o Google Colab).
3. Instalar dependencias:
   ```bash
   pip install pandas numpy matplotlib seaborn nltk wordcloud langdetect
   ```
4. Abrir y correr `Proyecyo2Analisis.ipynb` de principio a fin (o ejecutar `python eda_reto4.py`).
5. Los datasets limpios quedarán en `Datos/train_clean.csv` y `Datos/test_clean.csv`, y los gráficos en `figuras/`.

## Próximos pasos
- Vectorizar `body_lemmatized` con TF-IDF como modelo baseline (regresión logística).
- Evaluar un modelo basado en transformers (BERT/RoBERTa) usando `body` junto con `rule` y los ejemplos positivos/negativos como contexto.
- Incorporar `subreddit` y las variables derivadas (`url_count`, `has_legal_keyword`, etc.) como features adicionales al modelo de texto.
- Evaluar con la métrica oficial del reto (probabilidad de violación, AUC).

## Referencias
- [Jigsaw - Agile Community Rules Classification (Kaggle)](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)
- Guía del Proyecto 2 – CC3084 Data Science, UVG, Semestre II 2026.