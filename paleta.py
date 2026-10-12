"""
paleta.py  -  colores compartidos por la app (Plotly) y las figuras estáticas (matplotlib).

Validada con un validador de paletas categóricas (OKLCH/OKLab): luminosidad y croma dentro
de banda, separación >= 8 ΔE entre colores vecinos bajo daltonismo protan/deutan y
contraste >= 3:1 sobre fondo blanco.
"""
ORANGE = "#E8663D"        # naranja Reddit desaturado  -> "viola la regla"
PERI = "#6C8EEF"          # azul periwinkle             -> "no viola"
NEUTRAL = "#576F76"       # marcas sin identidad (p. ej. scatter con etiquetas directas)
GRID = "#EEF1F4"
INK = "#1C1C1C"
INK2 = "#4A5A63"
MUTED = "#7C8A92"

CLASS_COLORS = {"Viola": ORANGE, "No viola": PERI}

MODEL_COLORS = {          # orden validado; cada modelo conserva su color en todas las gráficas
    "SBERT + Reg. Logística": "#8B6FD9",
    "Random Forest": "#2F9E8F",
    "SVM": "#B5850F",
    "Regresión Logística": "#D9608F",
    "XGBoost": "#5E8F35",
    "Naive Bayes": "#4F86C6",
}

SEQ_SCALE = ["#F4F7FE", "#D3DEFB", "#A9BDF6", "#7F9CF0", "#5A78D6", "#3A55A8"]   # un solo tono


def model_color(name):
    return MODEL_COLORS.get(name, NEUTRAL)
