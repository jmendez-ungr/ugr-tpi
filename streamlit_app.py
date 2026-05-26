import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Predicción de monto de venta — TPI Ventas",
    page_icon="🛒",
    layout="wide"
)

SPREADSHEET_ID = "1mwzbJFK6c1Bkgu3_md0yxap8QcPJ569qxJ_xSjUpEjs"
GID = "0"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv&gid={GID}"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)

    text_cols = [
        "canal_venta", "region", "provincia", "sucursal", "segmento_cliente",
        "cliente_nuevo", "categoria_producto", "subcategoria_producto",
        "producto", "medio_pago", "mes"
    ]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    df["canal_venta"] = df["canal_venta"].replace({
        "web": "Web",
        "WEB": "Web",
        "Wpp": "WhatsApp"
    })

    df["medio_pago"] = df["medio_pago"].replace({
        "tarjeta de crédito": "Tarjeta de crédito",
        "Transfer.": "Transferencia"
    })

    df["fecha_venta"] = pd.to_datetime(df["fecha_venta"], errors="coerce", dayfirst=True)

    num_cols = [
        "precio_unitario", "cantidad", "descuento_pct", "costo_unitario",
        "tiempo_entrega_dias", "monto_total", "satisfaccion_cliente"
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["tiempo_entrega_dias"] = df["tiempo_entrega_dias"].fillna(df["tiempo_entrega_dias"].median())
    df["satisfaccion_cliente"] = df["satisfaccion_cliente"].fillna(df["satisfaccion_cliente"].median())

    df = df.drop_duplicates()

    df["anio"] = df["fecha_venta"].dt.year
    df["mes_num"] = df["fecha_venta"].dt.month
    df["dia_semana"] = df["fecha_venta"].dt.day_name()
    df["fin_de_semana"] = df["dia_semana"].isin(["Saturday", "Sunday"])

    df = df.sort_values("fecha_venta").reset_index(drop=True)
    return df

df = load_data()

@st.cache_resource
def train_model(dataframe):
    model = LinearRegression()
    X = dataframe[["cantidad"]]
    y = dataframe["monto_total"]
    model.fit(X, y)
    return model

model = train_model(df)

def format_money(value):
    return f"${value:,.0f}".replace(",", ".")

def classify_ticket(value):
    if value < 250000:
        return "Bajo"
    elif value < 700000:
        return "Medio"
    else:
        return "Alto"

def ticket_color(value):
    if value < 250000:
        return "#ef4444"
    elif value < 700000:
        return "#f59e0b"
    else:
        return "#22c55e"

def build_quick_read(quantity, discount, channel, pred):
    messages = []
    if quantity >= 5:
        messages.append("📦 Cantidad alta: principal impulsor del monto.")
    elif quantity <= 2:
        messages.append("📦 Cantidad baja: operación de volumen reducido.")
    if discount >= 20:
        messages.append("🏷️ Descuento elevado: reduce el total final de la venta.")
    elif discount == 0:
        messages.append("🏷️ Sin descuento: el valor final no tiene bonificación.")
    if channel in ["Web", "Marketplace"]:
        messages.append("🚚 Canal digital: puede implicar una logística más exigente.")
    elif channel == "Tienda física":
        messages.append("🏬 Tienda física: canal más directo y con menor espera de entrega.")
    if pred >= 700000:
        messages.append("💰 Ticket alto estimado: operación de valor relevante.")
    elif pred < 250000:
        messages.append("💰 Ticket bajo estimado: operación de valor acotado.")
    return messages

left, right = st.columns([4, 1.2])

with left:
    st.title("🛒 Predicción de monto de venta — TPI Ventas")
    st.caption("Proyecto demostrativo: cantidad + precio + descuento + canal → monto estimado")

with right:
    st.markdown(
        '''
        <div style="text-align:right; padding-top: 8px;">
            <span style="
                font-size:12px;
                border:1px solid #d1d5db;
                padding:6px 10px;
                border-radius:999px;
                color:#6b7280;">
                Streamlit · Demo TPI
            </span>
        </div>
        ''',
        unsafe_allow_html=True
    )

st.markdown("---")

c1, c2 = st.columns(2)

with c1:
    st.subheader("🧭 Introducción (formato proyecto)")
    st.markdown("**Objetivo de investigación**")
    st.markdown("""
- Analizar de manera simple el comportamiento de las ventas.
- Mostrar una primera aproximación al uso de un modelo lineal.
- Utilizar una herramienta complementaria e interactiva para la presentación.
""")

    st.markdown("**Origen de la fuente de datos**")
    st.markdown("""
- Dataset de ventas minoristas construido como proyecto modelo.
- Consultado desde Google Sheets.
- Incluye variables comerciales, logísticas y de satisfacción.
""")

    st.markdown("**Desarrollo del proyecto**")
    st.markdown("""
- Revisión inicial del dataset.
- Limpieza y preprocesamiento.
- EDA y visualizaciones.
- Regresión lineal simple con `cantidad` como explicativa.
""")

with c2:
    st.subheader("📌 Consideraciones")
    st.markdown("""
- El resultado del modelo es **orientativo**.
- Se prioriza la **interpretabilidad** y la **simplicidad**.
- La app no reemplaza el notebook ni el informe.
- Funciona como **complemento visual e interactivo** de la presentación.
""")

    st.markdown("**Alcance de esta app**")
    st.markdown("""
- Explorar filtros de la base.
- Visualizar resultados principales.
- Simular escenarios de venta.
- Ver cómo cambia la predicción al modificar inputs.
""")

st.markdown("---")

st.sidebar.title("Filtros del dataset")

canales = st.sidebar.multiselect(
    "Canal de venta",
    options=sorted(df["canal_venta"].dropna().unique()),
    default=sorted(df["canal_venta"].dropna().unique())
)

categorias = st.sidebar.multiselect(
    "Categoría",
    options=sorted(df["categoria_producto"].dropna().unique()),
    default=sorted(df["categoria_producto"].dropna().unique())
)

regiones = st.sidebar.multiselect(
    "Región",
    options=sorted(df["region"].dropna().unique()),
    default=sorted(df["region"].dropna().unique())
)

fecha_min = df["fecha_venta"].min().date()
fecha_max = df["fecha_venta"].max().date()

rango_fechas = st.sidebar.date_input(
    "Rango de fechas",
    value=(fecha_min, fecha_max),
    min_value=fecha_min,
    max_value=fecha_max
)

if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
    start_date, end_date = rango_fechas
else:
    start_date, end_date = fecha_min, fecha_max

df_f = df[
    (df["canal_venta"].isin(canales)) &
    (df["categoria_producto"].isin(categorias)) &
    (df["region"].isin(regiones)) &
    (df["fecha_venta"].dt.date >= start_date) &
    (df["fecha_venta"].dt.date <= end_date)
].copy()

if df_f.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Ventas filtradas", f"{df_f.shape[0]:,}".replace(",", "."))
k2.metric("Monto total", format_money(df_f["monto_total"].sum()))
k3.metric("Ticket promedio", format_money(df_f["monto_total"].mean()))
k4.metric("Satisfacción promedio", f"{df_f['satisfaccion_cliente'].mean():.2f}")

st.markdown("---")
st.subheader("⚙️ Interactivo")

b1, b2, b3, b4, b5 = st.columns([0.9, 1.1, 1.1, 1.3, 1.3])

with b1:
    st.metric("Registros filtrados", df_f.shape[0])

with b2:
    cantidad_input = st.slider("Cantidad", min_value=int(df["cantidad"].min()), max_value=int(df["cantidad"].max()), value=3, step=1)

with b3:
    descuento_input = st.slider("Descuento (%)", min_value=0, max_value=40, value=10, step=1)

with b4:
    canal_input = st.selectbox("Canal", options=sorted(df["canal_venta"].dropna().unique()), index=0)

with b5:
    precio_input = st.number_input("Precio unitario", min_value=1000.0, value=150000.0, step=1000.0)

st.caption(
    f"📝 Escenario interactivo: cantidad={cantidad_input} | descuento={descuento_input}% | "
    f"canal={canal_input} | precio unitario={format_money(precio_input)}"
)

st.markdown("---")

pred_modelo = model.predict(pd.DataFrame({"cantidad": [cantidad_input]}))[0]
monto_simulado = precio_input * cantidad_input * (1 - descuento_input / 100)
ticket_label = classify_ticket(monto_simulado)
color = ticket_color(monto_simulado)

r1, r2 = st.columns([1.35, 1])

with r1:
    st.subheader("Predicción / simulación de monto")
    st.markdown(
        f'''
        <div style="padding: 6px 0 0 0;">
            <div style="font-size: 42px; font-weight: 700; line-height: 1;">
                {format_money(monto_simulado)}
            </div>
            <div style="margin-top: 10px;">
                <span style="
                    background:{color}20;
                    color:{color};
                    border:1px solid {color}55;
                    padding:6px 10px;
                    border-radius:999px;
                    font-size:13px;
                    font-weight:600;">
                    Ticket {ticket_label}
                </span>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    progreso = min(max(monto_simulado / 1000000, 0), 1)
    st.progress(progreso)
    st.caption(
        f"Predicción del modelo lineal usando solo cantidad: {format_money(pred_modelo)} | "
        f"Simulación comercial con precio y descuento: {format_money(monto_simulado)}"
    )

with r2:
    st.subheader("Lectura rápida")
    mensajes = build_quick_read(cantidad_input, descuento_input, canal_input, monto_simulado)
    for msg in mensajes:
        st.markdown(f"- {msg}")

st.markdown("---")
st.subheader("📈 Relación entre cantidad y monto total")

x_line = np.arange(int(df["cantidad"].min()), int(df["cantidad"].max()) + 1)
y_line = model.predict(pd.DataFrame({"cantidad": x_line}))

fig_model = go.Figure()
fig_model.add_trace(go.Scatter(
    x=df_f["cantidad"], y=df_f["monto_total"],
    mode="markers", name="Datos filtrados",
    marker=dict(size=8, opacity=0.5)
))
fig_model.add_trace(go.Scatter(
    x=x_line, y=y_line,
    mode="lines", name="Recta de regresión",
    line=dict(width=3)
))
fig_model.add_trace(go.Scatter(
    x=[cantidad_input], y=[pred_modelo],
    mode="markers+text", name="Predicción modelo",
    marker=dict(size=14, symbol="diamond"),
    text=[f"Modelo: {format_money(pred_modelo)}"],
    textposition="top center"
))
fig_model.add_trace(go.Scatter(
    x=[cantidad_input], y=[monto_simulado],
    mode="markers+text", name="Simulación actual",
    marker=dict(size=14, symbol="circle"),
    text=[f"Simulación: {format_money(monto_simulado)}"],
    textposition="bottom center"
))
fig_model.update_layout(
    title="Recta de regresión + punto interactivo",
    xaxis_title="Cantidad",
    yaxis_title="Monto total",
    legend_title=""
)
st.plotly_chart(fig_model, use_container_width=True)

st.markdown("---")
st.subheader("📊 Distribuciones y contexto")

v1, v2 = st.columns(2)

with v1:
    fig_hist = px.histogram(
        df_f, x="monto_total", nbins=25,
        title="Distribución de monto total", marginal="box"
    )
    fig_hist.update_layout(xaxis_title="Monto total", yaxis_title="Frecuencia")
    st.plotly_chart(fig_hist, use_container_width=True)

with v2:
    fig_box = px.box(
        df_f, x="canal_venta", y="monto_total",
        color="canal_venta", title="Monto total por canal"
    )
    fig_box.update_layout(xaxis_title="Canal", yaxis_title="Monto total", showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

v3, v4 = st.columns(2)

with v3:
    ventas_mes = df_f.groupby("mes_num", as_index=False)["monto_total"].sum().sort_values("mes_num")
    fig_line = px.line(
        ventas_mes, x="mes_num", y="monto_total",
        markers=True, title="Evolución del monto total por mes"
    )
    fig_line.update_layout(xaxis_title="Mes", yaxis_title="Monto total acumulado")
    st.plotly_chart(fig_line, use_container_width=True)

with v4:
    corr_cols = ["precio_unitario", "cantidad", "descuento_pct", "costo_unitario", "tiempo_entrega_dias", "monto_total", "satisfaccion_cliente"]
    corr = df_f[corr_cols].corr()
    fig_heat = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        title="Heatmap de correlaciones",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")
st.subheader("🧾 Vista rápida del dataset filtrado")
st.dataframe(df_f.head(25), use_container_width=True)

st.caption(
    "Aplicación complementaria del TPI modelo. "
    "Las visualizaciones principales del informe siguen estando respaldadas por Python / Google Colab."
)
