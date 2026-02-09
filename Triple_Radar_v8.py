import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. SEGURIDAD ---
def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Control Maestro v8 - Acceso Restringido", type="password", on_change=password_entered, key="password")
        return False
    return st.session_state["password_correct"]

def password_entered():
    if st.session_state["password"] == "TU_CLAVE":
        st.session_state["password_correct"] = True
        del st.session_state["password"]
    else: st.session_state["password_correct"] = False

if not check_password(): st.stop()

st.set_page_config(page_title="Control Maestro v8", layout="wide", page_icon="💠")
st.title("🎛️ Control Maestro v8: Sistema de Comando Final")

# --- 2. CALCULADORA DE POSICIÓN INDEPENDIENTE (Sidebar) ---
st.sidebar.header("🛡️ CALCULADOR DE LOTES")
balance = st.sidebar.number_input("Capital Cuenta (USD)", value=1000.0)
riesgo_usd = st.sidebar.number_input("Riesgo a asumir (USD)", value=10.0)
pips_sl = st.sidebar.number_input("Pips de Stop Loss (SL)", min_value=1.0, value=20.0, step=1.0)

def calcular_lotes_final(riesgo, pips, activo):
    if pips == 0: return 0
    if "JPY" in activo:
        # USD/JPY: 1 lote standard, 1 pip (0.01) = ~$7.50 aprox
        return riesgo / (pips * 7.5) 
    else:
        # XAU/USD (Oro): 1 pip (0.10 usd) en 1 lote = $10
        return riesgo / (pips * 10)

# --- 3. DOBLE LEYENDA TÉCNICA ---
col_a, col_b = st.columns(2)
with col_a:
    with st.expander("📚 LEYENDA 1: GUÍA PRÁCTICA DEL ALGORITMO", expanded=True):
        st.markdown("""
        **Composición del Sistema:**
        * **VSA (Volume Spread Analysis):** Detecta anomalías entre volumen y rango de precio.
        * **Clustering K-Means:** Clasifica el mercado en estados de acumulación o tendencia.
        * **POC Dinámico:** Identifica el 'Muro Rojo' de liquidez institucional.
        * **VWAP Institucional:** El precio de equilibrio real usado por grandes fondos.
        """)
with col_b:
    with st.expander("🧠 LEYENDA 2: INTERPRETACIÓN DUMMIES VS PROS", expanded=True):
        st.markdown("""
        **Para Dummies:**
        * 🔴 **Muro Rojo:** Precio de control. No operes en contra de él.
        * 💠 **Diamante Azul:** Aviso de que los jefes están atrapando minoristas.
        
        **Para Profesionales:**
        * **Absorción:** Esfuerzo sin resultado (Volumen extremo con rango estrecho).
        * **Mean Reversion:** El precio tiende a regresar al VWAP cian.
        """)

# --- 4. GUÍA DE ESCENARIOS DE ALTA PROBABILIDAD ---
st.info("""
🔥 **ESCENARIOS A+ (Máxima Probabilidad):**
1. **El Rebote Institucional:** El precio toca el Muro Rojo (POC) + Aparece un Diamante Azul 💠.
2. **Capitulación:** Precio en extremo de tendencia + Diamante Azul 💠 (Indica regreso al VWAP 🔵).
""")

# --- 5. ANÁLISIS DE MERCADO ---
activos = {"Oro (Gold)": "GC=F", "Yen (USD/JPY)": "USDJPY=X"}
tfs = {"5m": "2d", "15m": "5d", "1h": "30d"}

for nombre, ticker in activos.items():
    st.markdown("---")
    try:
        # Cálculo de lotaje independiente
        lote_sugerido = calcular_lotes_final(riesgo_usd, pips_sl, ticker)
        
        c1, c2 = st.columns([3, 1])
        with c1: st.subheader(f"📊 {nombre}")
        with c2: st.success(f"**Lote Sugerido: {lote_sugerido:.2f}**")

        cols = st.columns(3)
        for idx, (tf, per) in enumerate(tfs.items()):
            df = yf.download(ticker, period=per, interval=tf, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # Cálculo del POC (Muro Rojo)
            bins = 20
            df['price_bin'] = pd.cut(df['Close'], bins=bins)
            poc_data = df.groupby('price_bin', observed=True)['Volume'].sum()
            idx_max = poc_data.idxmax()
            poc_price = (idx_max.left + idx_max.right) / 2
            
            # VWAP (Línea Cian)
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            # Lógica de Diamante v4
            df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['Range'] = df['High'] - df['Low']
            last_rvol = df['RVOL'].iloc[-1]
            last_range = df['Range'].iloc[-1]
            avg_range = df['Range'].rolling(20).mean().iloc[-1]
            es_diamante = (last_rvol > 2.0) and (last_range < avg_range)

            with cols[idx]:
                fig, ax = plt.subplots(figsize=(7, 5))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')
                
                ax.plot(df.index, df['Close'], color='white', alpha=0.3)
                ax.plot(df.index, df['VWAP'], color='cyan', linestyle='--', alpha=0.4)
                
                # POC: Marcado en eje y etiqueta flotante
                ax.axhline(y=poc_price, color='red', alpha=0.8, linewidth=1.5)
                ax.text(df.index[0], poc_price, f'  {poc_price:.2f}', color='red', fontweight='bold', va='bottom')
                
                # Inyección del nivel exacto en la escala (Eje Y)
                yticks = list(ax.get_yticks())
                if not any(abs(y - poc_price) < (df['Close'].max() - df['Close'].min()) * 0.05 for y in yticks):
                    yticks.append(poc_price)
                ax.set_yticks(yticks)

                if es_diamante:
                    ax.scatter(df.index[-1], df['Close'].iloc[-1], color='#00d4ff', s=200, marker='d', edgecolors='white')
                
                ax.set_title(f"TF: {tf}", color="white")
                ax.tick_params(colors='gray', labelsize=8)
                plt.xticks(rotation=45)
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Error procesando {nombre}: {e}")

st.caption("Control Maestro v8 | Precision & Execution Command Center")