import io
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import streamlit as st


st.set_page_config(page_title="Ajuste lineal con PDF", page_icon="📈", layout="wide")


# =========================
# Funciones auxiliares
# =========================

def parse_text_data(text: str) -> pd.DataFrame:
    """
    Convierte texto pegado por el usuario en un DataFrame con columnas x, y.
    Acepta separadores: coma, punto y coma, tabulador o espacios.
    Ignora líneas vacías y líneas que comienzan con #.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        normalized = line.replace(";", ",").replace("\t", ",")

        if "," in normalized:
            parts = [p.strip() for p in normalized.split(",") if p.strip()]
        else:
            parts = [p.strip() for p in normalized.split() if p.strip()]

        if len(parts) < 2:
            raise ValueError(
                f"No se pudieron leer dos columnas en la línea: '{line}'"
            )

        x_val = float(parts[0])
        y_val = float(parts[1])
        rows.append((x_val, y_val))

    if len(rows) < 2:
        raise ValueError("Se requieren al menos dos pares de datos (x, y).")

    return pd.DataFrame(rows, columns=["x", "y"])



def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Lee un CSV o TXT subido por el usuario.
    Si existen columnas llamadas x e y, las usa.
    En caso contrario, toma las primeras dos columnas.
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        # Intenta primero con separador flexible por regex.
        df = pd.read_csv(uploaded_file, sep=r"[,;\t\s]+", engine="python")

    if df.shape[1] < 2:
        raise ValueError("El archivo debe contener al menos dos columnas.")

    lower_map = {c.lower(): c for c in df.columns}

    if "x" in lower_map and "y" in lower_map:
        x_col = lower_map["x"]
        y_col = lower_map["y"]
        out = df[[x_col, y_col]].copy()
        out.columns = ["x", "y"]
    else:
        out = df.iloc[:, :2].copy()
        out.columns = ["x", "y"]

    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna().reset_index(drop=True)

    if len(out) < 2:
        raise ValueError("Después de limpiar el archivo, quedaron menos de dos filas válidas.")

    return out



def linear_regression_analysis(df: pd.DataFrame) -> dict:
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    n = len(df)

    if n < 3:
        raise ValueError("Se requieren al menos 3 puntos para reportar estadísticos del ajuste con mayor sentido.")

    result = stats.linregress(x, y)

    slope = result.slope
    intercept = result.intercept
    r_value = result.rvalue
    r2 = r_value**2
    p_value = result.pvalue
    slope_stderr = result.stderr
    intercept_stderr = result.intercept_stderr

    y_pred = intercept + slope * x
    residuals = y - y_pred

    dof = n - 2
    sse = np.sum(residuals**2)
    mse = sse / dof
    rmse = np.sqrt(mse)

    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)

    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_reg = np.sum((y_pred - y_mean) ** 2)

    f_stat = (ss_reg / 1) / (sse / dof) if sse > 0 else np.inf

    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha / 2, dof)

    slope_ci = (slope - t_crit * slope_stderr, slope + t_crit * slope_stderr)
    intercept_ci = (
        intercept - t_crit * intercept_stderr,
        intercept + t_crit * intercept_stderr,
    )

    return {
        "n": n,
        "x": x,
        "y": y,
        "slope": slope,
        "intercept": intercept,
        "r": r_value,
        "r2": r2,
        "p_value": p_value,
        "slope_stderr": slope_stderr,
        "intercept_stderr": intercept_stderr,
        "slope_ci": slope_ci,
        "intercept_ci": intercept_ci,
        "y_pred": y_pred,
        "residuals": residuals,
        "sse": sse,
        "mse": mse,
        "rmse": rmse,
        "f_stat": f_stat,
        "dof": dof,
        "x_mean": x_mean,
        "sxx": sxx,
    }



def build_main_figure(x, y, y_pred, slope, intercept, r2):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    order = np.argsort(x)

    ax.scatter(x, y, label="Datos experimentales")
    ax.plot(x[order], y_pred[order], label="Ajuste lineal")
    ax.set_title("Datos y ajuste lineal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()

    eq = f"y = {intercept:.6g} + {slope:.6g}x\nR² = {r2:.6f}"
    ax.text(
        0.03,
        0.97,
        eq,
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    fig.tight_layout()
    return fig



def build_residual_figure(y_pred, residuals):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(y_pred, residuals)
    ax.axhline(0, linestyle="--")
    ax.set_title("Gráfica de residuales")
    ax.set_xlabel("Valores ajustados")
    ax.set_ylabel("Residuales")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig



def analysis_text(stats_dict: dict) -> str:
    slope_ci = stats_dict["slope_ci"]
    intercept_ci = stats_dict["intercept_ci"]

    interpretation = []
    if stats_dict["p_value"] < 0.05:
        interpretation.append(
            "La pendiente es estadísticamente significativa a un nivel de confianza del 95%."
        )
    else:
        interpretation.append(
            "La pendiente no resulta estadísticamente significativa a un nivel de confianza del 95%."
        )

    if stats_dict["r2"] >= 0.90:
        interpretation.append("El ajuste lineal explica una proporción muy alta de la variabilidad de y.")
    elif stats_dict["r2"] >= 0.70:
        interpretation.append("El ajuste lineal explica una proporción alta de la variabilidad de y.")
    elif stats_dict["r2"] >= 0.50:
        interpretation.append("El ajuste lineal explica una proporción moderada de la variabilidad de y.")
    else:
        interpretation.append("El ajuste lineal explica una proporción limitada de la variabilidad de y.")

    return f"""
REPORTE DE AJUSTE LINEAL

Modelo:
    y = b0 + b1*x

Parámetros estimados:
    Intercepto (b0) = {stats_dict['intercept']:.8f}
    Pendiente  (b1) = {stats_dict['slope']:.8f}

Errores estándar:
    EE(b0) = {stats_dict['intercept_stderr']:.8f}
    EE(b1) = {stats_dict['slope_stderr']:.8f}

Intervalos de confianza al 95%:
    IC95%(b0) = [{intercept_ci[0]:.8f}, {intercept_ci[1]:.8f}]
    IC95%(b1) = [{slope_ci[0]:.8f}, {slope_ci[1]:.8f}]

Pruebas y bondad de ajuste:
    n            = {stats_dict['n']}
    gl           = {stats_dict['dof']}
    r            = {stats_dict['r']:.8f}
    R²           = {stats_dict['r2']:.8f}
    p(pendiente) = {stats_dict['p_value']:.8e}
    SSE          = {stats_dict['sse']:.8f}
    MSE          = {stats_dict['mse']:.8f}
    RMSE         = {stats_dict['rmse']:.8f}
    F            = {stats_dict['f_stat']:.8f}

Interpretación breve:
    - {interpretation[0]}
    - {interpretation[1]}
    - Revise la gráfica de residuales para detectar curvatura, heterocedasticidad o valores atípicos.
"""



def create_pdf_bytes(df: pd.DataFrame, stats_dict: dict) -> bytes:
    buffer = io.BytesIO()

    with PdfPages(buffer) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "Reporte de ajuste lineal"
        metadata["Author"] = "App Streamlit"
        metadata["Subject"] = "Regresión lineal simple"
        metadata["CreationDate"] = datetime.now()

        # Página 1: Datos y ajuste
        fig1 = build_main_figure(
            stats_dict["x"],
            stats_dict["y"],
            stats_dict["y_pred"],
            stats_dict["slope"],
            stats_dict["intercept"],
            stats_dict["r2"],
        )
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        # Página 2: Residuales
        fig2 = build_residual_figure(stats_dict["y_pred"], stats_dict["residuals"])
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        # Página 3: Resumen estadístico
        fig3, ax3 = plt.subplots(figsize=(8.5, 11))
        ax3.axis("off")
        ax3.text(
            0.03,
            0.97,
            analysis_text(stats_dict),
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
        )
        fig3.tight_layout()
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

        # Página 4: Tabla de datos
        fig4, ax4 = plt.subplots(figsize=(8.5, 11))
        ax4.axis("off")
        ax4.set_title("Datos experimentales", fontsize=14, pad=14)

        table_df = df.copy()
        table_df.index = np.arange(1, len(table_df) + 1)
        table = ax4.table(
            cellText=np.round(table_df.values, 6),
            colLabels=table_df.columns,
            rowLabels=table_df.index,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.35)
        fig4.tight_layout()
        pdf.savefig(fig4, bbox_inches="tight")
        plt.close(fig4)

    buffer.seek(0)
    return buffer.getvalue()


# =========================
# Interfaz
# =========================

st.title("📈 Ajuste lineal con generación de PDF")
st.write(
    "Capture o cargue pares de datos x,y, calcule la regresión lineal simple y descargue un reporte en PDF."
)

with st.expander("Ver formato de entrada manual"):
    st.code(
        """1, 2.1
2, 4.0
3, 5.9
4, 8.2
5, 10.1""",
        language="text",
    )

col1, col2 = st.columns([1.2, 1])

with col1:
    input_mode = st.radio(
        "Método de entrada",
        options=["Pegar datos", "Subir archivo CSV/TXT"],
        horizontal=True,
    )

    if input_mode == "Pegar datos":
        raw_text = st.text_area(
            "Pegue aquí los datos x,y (una fila por punto)",
            height=220,
            value="1, 2.1\n2, 4.0\n3, 5.9\n4, 8.2\n5, 10.1\n6, 12.2\n7, 13.8\n8, 16.1",
        )
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader(
            "Suba un archivo CSV o TXT",
            type=["csv", "txt"],
            help="Puede usar columnas llamadas x e y o simplemente dejar x en la primera columna y y en la segunda.",
        )
        raw_text = None

    calcular = st.button("Calcular ajuste", type="primary")

with col2:
    st.subheader("Qué incluye el reporte")
    st.markdown(
        """
- Ecuación de la recta ajustada
- Parámetros estadísticos básicos
- Intervalos de confianza al 95%
- Gráfica de datos con ajuste
- Gráfica de residuales
- Tabla de datos experimentales
        """
    )


if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "stats_dict" not in st.session_state:
    st.session_state.stats_dict = None
if "df_data" not in st.session_state:
    st.session_state.df_data = None


if calcular:
    try:
        if input_mode == "Pegar datos":
            df = parse_text_data(raw_text)
        else:
            if uploaded_file is None:
                raise ValueError("Debe subir un archivo antes de calcular.")
            df = read_uploaded_file(uploaded_file)

        if len(df) < 3:
            raise ValueError("Se requieren al menos 3 puntos válidos.")

        stats_dict = linear_regression_analysis(df)
        pdf_bytes = create_pdf_bytes(df, stats_dict)

        st.session_state.df_data = df
        st.session_state.stats_dict = stats_dict
        st.session_state.pdf_bytes = pdf_bytes

        st.success("Ajuste calculado correctamente. Ya puede revisar resultados y descargar el PDF.")

    except Exception as e:
        st.session_state.df_data = None
        st.session_state.stats_dict = None
        st.session_state.pdf_bytes = None
        st.error(f"Ocurrió un error: {e}")


if st.session_state.df_data is not None and st.session_state.stats_dict is not None:
    df = st.session_state.df_data
    stats_dict = st.session_state.stats_dict

    st.divider()
    st.subheader("Datos leídos")
    st.dataframe(df, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pendiente", f"{stats_dict['slope']:.6g}")
    m2.metric("Intercepto", f"{stats_dict['intercept']:.6g}")
    m3.metric("R²", f"{stats_dict['r2']:.6f}")
    m4.metric("p(pendiente)", f"{stats_dict['p_value']:.3e}")

    g1, g2 = st.columns(2)
    with g1:
        fig_main = build_main_figure(
            stats_dict["x"],
            stats_dict["y"],
            stats_dict["y_pred"],
            stats_dict["slope"],
            stats_dict["intercept"],
            stats_dict["r2"],
        )
        st.pyplot(fig_main)
        plt.close(fig_main)

    with g2:
        fig_res = build_residual_figure(stats_dict["y_pred"], stats_dict["residuals"])
        st.pyplot(fig_res)
        plt.close(fig_res)

    st.subheader("Resumen estadístico")
    summary_df = pd.DataFrame(
        {
            "Parámetro": [
                "n",
                "Pendiente",
                "Intercepto",
                "r",
                "R²",
                "Error estándar de pendiente",
                "Error estándar de intercepto",
                "IC95% pendiente (inferior)",
                "IC95% pendiente (superior)",
                "IC95% intercepto (inferior)",
                "IC95% intercepto (superior)",
                "SSE",
                "MSE",
                "RMSE",
                "F",
                "p(pendiente)",
            ],
            "Valor": [
                stats_dict["n"],
                stats_dict["slope"],
                stats_dict["intercept"],
                stats_dict["r"],
                stats_dict["r2"],
                stats_dict["slope_stderr"],
                stats_dict["intercept_stderr"],
                stats_dict["slope_ci"][0],
                stats_dict["slope_ci"][1],
                stats_dict["intercept_ci"][0],
                stats_dict["intercept_ci"][1],
                stats_dict["sse"],
                stats_dict["mse"],
                stats_dict["rmse"],
                stats_dict["f_stat"],
                stats_dict["p_value"],
            ],
        }
    )
    st.dataframe(summary_df, use_container_width=True)


if st.session_state.pdf_bytes is not None:
    st.divider()
    st.download_button(
        label="Descargar reporte PDF",
        data=st.session_state.pdf_bytes,
        file_name="reporte_ajuste_lineal.pdf",
        mime="application/pdf",
    )


st.divider()
st.caption(
    "Sugerencia: si va a analizar datos experimentales reales, revise linealidad, homocedasticidad y presencia de valores atípicos antes de interpretar el modelo."
)
