
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# -----------------------------
# Configuración general
# -----------------------------
st.set_page_config(
    page_title="TPI Ventas - Dashboard Interactivo",
    page_icon="📊",
    layout="wide"
)

SPREADSHEET_ID = "1mwzbJFK6c1Bkgu3_md0yxap8QcPJ569qxJ_xSjUpEjs"
GID = "0"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv&gid={GID}"

# -----------------------------
# Carga y limpieza
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)

    # Limpieza base
    text_cols = [
        'canal_venta', 'region', 'provincia', 'sucursal', 'segmento_cliente',
        'cliente_nuevo', 'categoria_producto', 'subcategoria_producto',
        'producto', 'medio_pago', 'mes'
    ]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    df['canal_venta'] = df['canal_venta'].replace({
        'web': 'Web',
        'WEB': 'Web',
        'Wpp': 'WhatsApp'
    })

    df['medio_pago'] = df['medio_pago'].replace({
        'tarjeta de crédito': 'Tarjeta de crédito',
        'Transfer.': 'Transferencia'
    })

    df['fecha_venta'] = pd.to_datetime(df['fecha_venta'], errors='coerce', dayfirst=True)

    num_cols = [
        'precio_unitario', 'cantidad', 'descuento_pct', 'costo_unitario',
        'tiempo_entrega_dias', 'monto_total', 'satisfaccion_cliente'
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['tiempo_entrega_dias'] = df['tiempo_entrega_dias'].fillna(df['tiempo_entrega_dias'].median())
    df['satisfaccion_cliente'] = df['satisfaccion_cliente'].fillna(df['satisfaccion_cliente'].median())

    df = df.drop_duplicates()

    df['anio'] = df['fecha_venta'].dt.year
    df['mes_num'] = df['fecha_venta'].dt.month
    df['dia_semana'] = df['fecha_venta'].dt.day_name()
    df['fin_de_semana'] = df['dia_semana'].isin(['Saturday', 'Sunday'])

    # Orden útil
    df = df.sort_values('fecha_venta').reset_index(drop=True)
    return df

df = load_data()

# -----------------------------
# Modelo simple para interacción
# -----------------------------
@st.cache_resource
def train_model(dataframe):
    model = LinearRegression()
    X = dataframe[['cantidad']]
    y = dataframe['monto_total']
    model.fit(X, y)
    return model

model = train_model(df)

# -----------------------------
# Sidebar filtros
# -----------------------------
st.sidebar.title("Filtros")
st.sidebar.caption("Podés usar estos filtros para explorar el dataset del proyecto modelo.")

canales = st.sidebar.multiselect(
    "Canal de venta",
    options=sorted(df['canal_venta'].dropna().unique()),
    default=sorted(df['canal_venta'].dropna().unique())
)

categorias = st.sidebar.multiselect(
    "Categoría",
    options=sorted(df['categoria_producto'].dropna().unique()),
    default=sorted(df['categoria_producto'].dropna().unique())
)

regiones = st.sidebar.multiselect(
    "Región",
    options=sorted(df['region'].dropna().unique()),
    default=sorted(df['region'].dropna().unique())
)

fecha_min = df['fecha_venta'].min().date()
fecha_max = df['fecha_venta'].max().date()
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
    (df['canal_venta'].isin(canales)) &
    (df['categoria_producto'].isin(categorias)) &
    (df['region'].isin(regiones)) &
    (df['fecha_venta'].dt.date >= start_date) &
    (df['fecha_venta'].dt.date <= end_date)
].copy()

# -----------------------------
# Header
# -----------------------------
st.title("📊 TPI - Dashboard Interactivo de Ventas")
st.markdown(
    """
    Este dashboard funciona como **herramienta complementaria de presentación** del proyecto modelo.
    Permite explorar la base, visualizar resultados y probar una pequeña interacción con el modelo lineal.
    """
)

if df_f.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

# -----------------------------
# KPIs
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Ventas filtradas", f"{df_f.shape[0]:,}".replace(",", "."))
k2.metric("Monto total", f"${df_f['monto_total'].sum():,.0f}".replace(",", "."))
k3.metric("Ticket promedio", f"${df_f['monto_total'].mean():,.0f}".replace(",", "."))
k4.metric("Satisfacción promedio", f"{df_f['satisfaccion_cliente'].mean():.2f}")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Resumen general",
    "Visualizaciones",
    "Exploración detallada",
    "Modelo interactivo"
])

with tab1:
    st.subheader("Resumen general del dataset filtrado")

    c1, c2 = st.columns([1.2, 1])

    with c1:
        ventas_mes = (
            df_f.groupby('mes_num', as_index=False)['monto_total']
            .sum()
            .sort_values('mes_num')
        )
        fig_line = px.line(
            ventas_mes,
            x='mes_num',
            y='monto_total',
            markers=True,
            title="Evolución del monto total por mes"
        )
        fig_line.update_layout(xaxis_title="Mes", yaxis_title="Monto total acumulado")
        st.plotly_chart(fig_line, use_container_width=True)

    with c2:
        monto_categoria = (
            df_f.groupby('categoria_producto', as_index=False)['monto_total']
            .sum()
            .sort_values('monto_total', ascending=False)
        )
        fig_bar = px.bar(
            monto_categoria,
            x='categoria_producto',
            y='monto_total',
            title="Monto total por categoría",
            text_auto='.2s'
        )
        fig_bar.update_layout(xaxis_title="Categoría", yaxis_title="Monto total")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Vista rápida de los datos")
    st.dataframe(df_f.head(20), use_container_width=True)

with tab2:
    st.subheader("Visualizaciones principales")

    v1, v2 = st.columns(2)

    with v1:
        fig_hist = px.histogram(
            df_f,
            x='monto_total',
            nbins=25,
            title="Distribución de monto total",
            marginal="box"
        )
        fig_hist.update_layout(xaxis_title="Monto total", yaxis_title="Frecuencia")
        st.plotly_chart(fig_hist, use_container_width=True)

    with v2:
        fig_box = px.box(
            df_f,
            x='canal_venta',
            y='monto_total',
            color='canal_venta',
            title="Monto total por canal de venta"
        )
        fig_box.update_layout(xaxis_title="Canal", yaxis_title="Monto total", showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    v3, v4 = st.columns(2)

    with v3:
        fig_scatter = px.scatter(
            df_f,
            x='cantidad',
            y='monto_total',
            color='categoria_producto',
            hover_data=['producto', 'canal_venta', 'region'],
            title="Relación entre cantidad y monto total"
        )
        fig_scatter.update_layout(xaxis_title="Cantidad", yaxis_title="Monto total")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with v4:
        corr_cols = ['precio_unitario', 'cantidad', 'descuento_pct', 'costo_unitario', 'tiempo_entrega_dias', 'monto_total', 'satisfaccion_cliente']
        corr = df_f[corr_cols].corr()
        fig_heat = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Heatmap de correlaciones",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    st.subheader("Exploración detallada")

    d1, d2 = st.columns([1, 1])

    with d1:
        resumen_canal = (
            df_f.groupby('canal_venta')['monto_total']
            .agg(['count', 'sum', 'mean', 'median'])
            .reset_index()
            .sort_values('sum', ascending=False)
        )
        st.markdown("#### Resumen por canal")
        st.dataframe(resumen_canal, use_container_width=True)

    with d2:
        resumen_categoria = (
            df_f.groupby('categoria_producto')['monto_total']
            .agg(['count', 'sum', 'mean', 'median'])
            .reset_index()
            .sort_values('sum', ascending=False)
        )
        st.markdown("#### Resumen por categoría")
        st.dataframe(resumen_categoria, use_container_width=True)

    st.markdown("#### Tabla dinámica de apoyo")
    pivot = pd.pivot_table(
        df_f,
        values='monto_total',
        index='region',
        columns='canal_venta',
        aggfunc='sum',
        fill_value=0
    )
    st.dataframe(pivot, use_container_width=True)

with tab4:
    st.subheader("Modelo interactivo")
    st.markdown(
        """
        En el notebook del proyecto se ajusta una **regresión lineal simple** para explicar `monto_total`
        a partir de `cantidad`. Acá podés mover un control y ver cómo cambia la predicción.
        """
    )

    m1, m2 = st.columns([1, 1.3])

    with m1:
        cantidad_input = st.slider(
            "Elegí una cantidad de unidades",
            min_value=int(df['cantidad'].min()),
            max_value=int(df['cantidad'].max()),
            value=3,
            step=1
        )

        pred = model.predict(pd.DataFrame({'cantidad': [cantidad_input]}))[0]
        st.metric("Monto total estimado", f"${pred:,.0f}".replace(",", "."))
        st.caption("Predicción generada por el modelo lineal simple entrenado con toda la base limpia.")

        st.markdown("#### Parámetros del modelo")
        st.write(f"**Intercepto:** {model.intercept_:,.2f}".replace(",", "."))
        st.write(f"**Coeficiente de cantidad:** {model.coef_[0]:,.2f}".replace(",", "."))

    with m2:
        x_line = np.arange(int(df['cantidad'].min()), int(df['cantidad'].max()) + 1)
        y_line = model.predict(pd.DataFrame({'cantidad': x_line}))

        fig_model = go.Figure()
        fig_model.add_trace(go.Scatter(
            x=df['cantidad'],
            y=df['monto_total'],
            mode='markers',
            name='Datos reales',
            marker=dict(size=8, opacity=0.55)
        ))
        fig_model.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Recta de regresión',
            line=dict(width=3)
        ))
        fig_model.add_trace(go.Scatter(
            x=[cantidad_input],
            y=[pred],
            mode='markers+text',
            name='Valor seleccionado',
            marker=dict(size=14, symbol='diamond'),
            text=[f"Cantidad: {cantidad_input}<br>Pred: ${pred:,.0f}".replace(",", ".")],
            textposition='top center'
        ))
        fig_model.update_layout(
            title="Modelo lineal interactivo: cantidad vs monto_total",
            xaxis_title="Cantidad",
            yaxis_title="Monto total"
        )
        st.plotly_chart(fig_model, use_container_width=True)

    st.markdown("#### Simulador simple de venta")
    st.markdown("Además del modelo, podés probar una simulación comercial básica con precio, cantidad y descuento.")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        precio = st.number_input("Precio unitario", min_value=1000.0, value=150000.0, step=1000.0)
    with s2:
        cant = st.number_input("Cantidad", min_value=1, value=2, step=1)
    with s3:
        descuento = st.slider("Descuento %", min_value=0, max_value=40, value=10, step=1)
    with s4:
        monto_simulado = precio * cant * (1 - descuento/100)
        st.metric("Monto simulado", f"${monto_simulado:,.0f}".replace(",", "."))

st.divider()
st.caption("Dashboard complementario del TPI modelo. Las visualizaciones del informe principal siguen estando respaldadas por Python / Google Colab.")
