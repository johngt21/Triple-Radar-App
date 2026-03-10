import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURACIÓN DE PÁGINA ────────────────────────────────────────────────
st.set_page_config(page_title="TRIPLE RADAR v10.0", layout="wide", page_icon="🎯")

# ─── 1. SEGURIDAD ────────────────────────────────────────────────────────────
def check_password():
    if "password_correct" not in st.session_state:
        st.text_input(
            "TRIPLE RADAR v10.0 — Acceso Restringido",
            type="password",
            on_change=password_entered,
            key="password"
        )
        return False
    return st.session_state["password_correct"]

def password_entered():
    if st.session_state["password"] == "TU_CLAVE":
        st.session_state["password_correct"] = True
        del st.session_state["password"]
    else:
        st.session_state["password_correct"] = False

if not check_password():
    st.stop()

st.title("🎯 TRIPLE RADAR v10.0 — ARIMA · GARCH · RSI Divergence · POC · VWAP")

# ─── 2. MANUAL TÉCNICO ───────────────────────────────────────────────────────
with st.expander("📖 MANUAL TÉCNICO COMPLETO — Leer antes de operar", expanded=False):
    st.markdown("""
## 🎯 TRIPLE RADAR v10.0 — Manual Técnico

### ACTIVOS MONITOREADOS
| Activo | Ticker | Descripción |
|--------|--------|-------------|
| **Oro (Gold Futures)** | GC=F | Contrato de futuros del oro al contado. Activo de refugio y cobertura inflacionaria. Muy sensible a tasas reales y dólar. |
| **Nasdaq 100 (NQ Futures)** | NQ=F | Contrato de futuros del índice tecnológico Nasdaq 100. Alta correlación con apetito de riesgo y liquidez global. |
| **Bitcoin** | BTC-USD | Criptoactivo de mayor capitalización de mercado. Opera 24/7. Alta volatilidad. Correlación creciente con activos de riesgo. |

### TEMPORALIDADES ANALIZADAS
| Timeframe | Propósito técnico | Datos utilizados |
|-----------|-------------------|-----------------|
| **1H** | Ejecución táctica intradiaria. Señales de entrada/salida de corto plazo. | Últimos 7 días en velas de 1 hora. |
| **4H** | Confirmación de tendencia intermedia. Reduce ruido del 1H. | 60 días de datos 1H resampleados a 4H. |
| **1D** | Marco estructural. Define bias direccional de fondo. | 1 año en velas diarias. ⚠️ El último dato corresponde al cierre del día previo. |

---

### INDICADORES POR GRÁFICO

#### 🔴 POC — Point of Control (Línea Roja Horizontal)
**Definición técnica:** El nivel de precio con mayor volumen negociado en el período analizado. Deriva del concepto de *Volume Profile* (perfil de volumen vertical).
**¿Por qué es relevante?** El mercado pasa más tiempo negociando alrededor del POC porque representa la zona de *fair value* de corto plazo donde compradores y vendedores están más equilibrados.
**Para no financieros:** Es el precio "más justo" donde más gente compró y vendió. El precio suele volver a él, como un imán.
**Señal operativa:**
- Precio sostenido **por encima del POC** → presión alcista estructural.
- Precio sostenido **por debajo del POC** → presión bajista estructural.
- Ruptura del POC **con volumen creciente** → señal de continuación fuerte.
- Precio oscilando **alrededor del POC** → mercado en consolidación/indecisión.

---

#### 🩵 VWAP — Volume-Weighted Average Price (Línea Cyan)
**Definición técnica:** Precio promedio de todas las transacciones del período, ponderado por el volumen de cada operación. Fórmula: VWAP = Σ(Precio × Volumen) / Σ(Volumen).
**¿Por qué es relevante?** Es la referencia de ejecución institucional más utilizada en el mundo. Fondos, *market makers* y algoritmos la usan como *benchmark*: comprar por debajo del VWAP = buena ejecución; vender por encima = buena ejecución.
**Para no financieros:** Es el precio promedio "real" del mercado. Si el precio está arriba, los grandes compraron barato y el mercado va alcista. Si está abajo, los grandes vendieron caro y el mercado va bajista.
**Señal operativa:**
- Precio **por encima del VWAP** → sesgo alcista institucional. Favorece posiciones largas.
- Precio **por debajo del VWAP** → sesgo bajista institucional. Favorece posiciones cortas.
- **Cruce del VWAP** → cambio de estructura de mercado. Señal de alta prioridad.

---

#### 💎 DIAMANTE — Señal de Compresión de Volatilidad con Volumen Anómalo
**Definición técnica:** Se activa cuando se cumplen simultáneamente: (1) RVOL (*Relative Volume*) > 2.0x la media de 20 períodos, y (2) el rango de la vela (High - Low) es menor al rango promedio de 20 velas.
**¿Por qué es relevante?** Este patrón indica que hay actividad transaccional inusualmente alta pero el precio no se mueve. Eso sugiere acumulación silenciosa (*absorption*) o distribución institucional. Precede expansiones de volatilidad.
**Para no financieros:** Mucho movimiento "invisible" de dinero con precio quieto. Cuando el dinero grande se mueve sin que el precio se mueva, algo está a punto de pasar.
**Señal operativa:**
- Diamante azul en la última vela → posible *breakout* inminente. La **dirección** la define la siguiente vela.
- Combinar siempre con la posición del precio respecto al POC y VWAP para filtrar la dirección probable.

---

#### 📊 RSI DIVERGENCE — Índice de Fuerza Relativa con Detección de Divergencias (14 períodos)
**Definición técnica:** El RSI (Wilder, 1978) mide el momentum relativo del precio en escala 0-100. Se calculan las ganancias y pérdidas promedio de los últimos 14 períodos. Se detectan divergencias algorítmicamente comparando máximos/mínimos del precio vs. máximos/mínimos del RSI.

**Zonas del RSI:**
| Zona | Valor | Interpretación |
|------|-------|----------------|
| Sobrecompra | > 70 | El activo ha subido rápido; aumenta probabilidad de corrección. |
| Zona neutral | 50-70 | Momentum alcista moderado. |
| Zona neutral | 30-50 | Momentum bajista moderado. |
| Sobreventa | < 30 | El activo ha bajado rápido; aumenta probabilidad de rebote. |

**Tipos de divergencia:**
- 🟢 **Divergencia Alcista** (triángulo verde ▲): El precio forma un mínimo más bajo (LL) **pero** el RSI forma un mínimo más alto (HL). El momentum bajista se agota. Señal de posible reversión al alza.
- 🔴 **Divergencia Bajista** (triángulo rojo ▼): El precio forma un máximo más alto (HH) **pero** el RSI forma un máximo más bajo (LH). El momentum alcista se agota. Señal de posible reversión a la baja.
**Para no financieros:** El motor (RSI) y el coche (precio) van en direcciones opuestas. Cuando eso pasa, el coche suele corregir para alinearse con el motor.
**Señal operativa:** Las divergencias en 4H y 1D tienen mayor peso estadístico que en 1H.

---

### MODELOS PREDICTIVOS CUANTITATIVOS

#### 📈 ARIMA — AutoRegressive Integrated Moving Average (2,1,2)
**Definición técnica:** Modelo econométrico de series de tiempo que descompone el precio en tres componentes: AR(2) = el precio depende de los 2 precios anteriores; I(1) = se diferencia una vez para estacionalizar la serie; MA(2) = se modela el error de los 2 períodos anteriores.
**Para no financieros:** Una fórmula matemática que aprende el patrón histórico del precio y extrapola hacia dónde debería ir. Como una regresión lineal avanzada con memoria.
**Señal operativa:**
- Línea verde punteada ascendente + bandas azul cielo **estrechas** → tendencia alcista con alta confianza estadística. Mejor momento para considerar compra.
- Línea descendente + bandas **estrechas** → tendencia bajista confirmada. Mejor momento para considerar venta.
- Bandas **muy anchas** → alta incertidumbre estadística. Reducir tamaño de posición o esperar.
- Las bandas representan el intervalo de confianza al **95%**: el precio debería estar dentro de ese rango el 95% del tiempo si el modelo es correcto.

#### 📉 GARCH — Generalized AutoRegressive Conditional Heteroskedasticity (1,1)
**Definición técnica:** Modelo econométrico para predicción de volatilidad condicional. Captura el fenómeno de *volatility clustering*: períodos de alta volatilidad tienden a seguir a períodos de alta volatilidad (y viceversa). Modela la varianza condicional del retorno.
**Para no financieros:** Predice qué tan "agitado" va a estar el mercado en los próximos períodos. No dice para dónde va el precio, sino cuánto puede moverse.
**Niveles de volatilidad predicha:**
| Nivel | Criterio | Acción sugerida |
|-------|----------|-----------------|
| 🔵 BAJA | < 50% de la media histórica | Momento óptimo para estrategias de *breakout*. Mercado comprimido. |
| 🟢 NORMAL | 50-100% de la media | Condiciones estándar de operación. |
| 🟠 ALTA | 100-150% de la media | Ampliar stops. Reducir tamaño de posición. |
| 🔴 MUY ALTA | > 150% de la media | Riesgo elevado. Considerar salir o no entrar nuevas posiciones. |

---

### 🔥 FUEGO MAESTRO — Señal de Confluencia Multi-Temporal
**Definición técnica:** Evaluación del sesgo direccional (precio vs. VWAP) en los 3 timeframes (1H, 4H, 1D) de forma simultánea.
**¿Por qué es relevante?** La confluencia multi-temporal reduce significativamente los falsos positivos. Cuando el 1H, 4H y 1D apuntan en la misma dirección, la probabilidad estadística de continuación aumenta considerablemente.

| Señal | Condición | Interpretación |
|-------|-----------|----------------|
| 🔥🔥🔥 FUEGO MAESTRO ALCISTA | 3/3 TF con precio > VWAP | Máxima alineación alcista. Mayor probabilidad de continuación. |
| 🧊🧊🧊 ALINEACIÓN BAJISTA TOTAL | 3/3 TF con precio < VWAP | Máxima alineación bajista. Mayor probabilidad de continuación. |
| ⬆️ SESGO ALCISTA PARCIAL | 2/3 TF alcistas | Tendencia alcista en desarrollo. Esperar confirmación del TF discordante. |
| ⬇️ SESGO BAJISTA PARCIAL | 2/3 TF bajistas | Tendencia bajista en desarrollo. Esperar confirmación del TF discordante. |
| ⚖️ MERCADO MIXTO | 1/3 o empate | Conflicto entre timeframes. No operar hasta alineación. |

---

### ⚡ PROTOCOLO DE ANÁLISIS RECOMENDADO (top-down)
1. **GARCH primero** → ¿Cuál es el régimen de volatilidad actual? Define tamaño de posición y amplitud de stops.
2. **ARIMA** → ¿Cuál es la dirección estadísticamente esperada? Define el sesgo direccional del análisis.
3. **Fuego Maestro (1D → 4H → 1H)** → ¿Los 3 TF confirman la dirección del ARIMA?
4. **POC en 4H y 1D** → Identifica los niveles estructurales clave para entrada y stop loss.
5. **RSI Divergence en 1H** → Timing fino para la entrada táctica.
6. **Diamante** → Confirmación final de presión compradora/vendedora silenciosa.
7. **Sin stop loss = sin operación.** Este sistema indica probabilidad estadística, no certeza.

> ⚠️ *TRIPLE RADAR es una herramienta de análisis cuantitativo. No constituye asesoramiento financiero. El trading implica riesgo de pérdida de capital.*
""")

# ─── 3. IMPORTS OPCIONALES ───────────────────────────────────────────────────
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_DISPONIBLE = True
except ImportError:
    ARIMA_DISPONIBLE = False
    st.warning("⚠️ statsmodels no instalado. ARIMA no disponible. Ejecutar: pip install statsmodels")

try:
    from arch import arch_model
    GARCH_DISPONIBLE = True
except ImportError:
    GARCH_DISPONIBLE = False
    st.warning("⚠️ arch no instalado. GARCH no disponible. Ejecutar: pip install arch")

# ─── 4. FUNCIONES AUXILIARES ─────────────────────────────────────────────────

def calcular_rsi(series, periodo=14):
    """RSI estándar de Wilder (14 períodos)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=periodo).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=periodo).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def detectar_divergencias(precios, rsi, lookback=5):
    """
    Detección algorítmica de divergencias RSI-Precio.
    Retorna índices de divergencias alcistas y bajistas.
    """
    divergencias_alcistas = []
    divergencias_bajistas = []
    n = len(precios)
    for i in range(lookback * 2, n):
        p_curr = float(precios.iloc[i])
        p_prev = float(precios.iloc[i - lookback])
        r_curr = float(rsi.iloc[i])   if not pd.isna(rsi.iloc[i])   else np.nan
        r_prev = float(rsi.iloc[i - lookback]) if not pd.isna(rsi.iloc[i - lookback]) else np.nan
        if np.isnan(r_curr) or np.isnan(r_prev):
            continue
        # Divergencia bajista: precio HH, RSI LH (señal de agotamiento alcista)
        if p_curr > p_prev and r_curr < r_prev and r_curr > 60:
            divergencias_bajistas.append(i)
        # Divergencia alcista: precio LL, RSI HL (señal de agotamiento bajista)
        if p_curr < p_prev and r_curr > r_prev and r_curr < 40:
            divergencias_alcistas.append(i)
    return divergencias_alcistas, divergencias_bajistas


def resamplear_4h(df_1h):
    """Resamplea OHLCV de 1H a 4H."""
    if df_1h.empty:
        return pd.DataFrame()
    try:
        df_4h = df_1h[['Open', 'High', 'Low', 'Close', 'Volume']].resample('4h').agg({
            'Open':   'first',
            'High':   'max',
            'Low':    'min',
            'Close':  'last',
            'Volume': 'sum'
        }).dropna(subset=['Close'])
        return df_4h
    except Exception:
        return pd.DataFrame()


def graficar_arima_forecast(precios, nombre_activo, periodos=10):
    """ARIMA(2,1,2): predicción de precio con IC al 95%."""
    if not ARIMA_DISPONIBLE:
        return None, None
    try:
        s = precios.dropna().astype(float).tail(200)
        if len(s) < 50:
            return None, None
        fit = ARIMA(s, order=(2, 1, 2)).fit()
        fc  = fit.get_forecast(steps=periodos)
        mu  = fc.predicted_mean
        ci  = fc.conf_int(alpha=0.05)
        freq = pd.infer_freq(s.index[-20:]) or 'h'
        idx_fut = pd.date_range(start=s.index[-1], periods=periodos + 1, freq=freq)[1:]
        hist = s.tail(50)

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.plot(hist.index, hist.values, color='white', linewidth=1.5, label='Precio real')
        ax.plot(idx_fut, mu.values, color='lime', linewidth=2, linestyle='--', label='Predicción ARIMA')
        ax.fill_between(idx_fut, ci.iloc[:, 0].values, ci.iloc[:, 1].values,
                        color='skyblue', alpha=0.35, label='IC 95%')
        ax.plot([hist.index[-1], idx_fut[0]], [hist.values[-1], mu.values[0]],
                color='lime', linewidth=1.5, linestyle='--')

        p0 = float(s.iloc[-1])
        pf = float(mu.iloc[-1])
        pct = (pf - p0) / p0 * 100
        dir_txt = "▲ SUBE" if pf > p0 else "▼ BAJA"
        dir_col = 'lime' if pf > p0 else 'tomato'

        ax.set_title(f"ARIMA — {nombre_activo}   {dir_txt}  {pct:+.2f}%",
                     color=dir_col, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', colors='gray', labelsize=7, rotation=45)
        ax.tick_params(axis='y', colors='gray', labelsize=8)
        ax.legend(loc='upper left', fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax.grid(color='gray', linestyle=':', linewidth=0.3, alpha=0.5)
        plt.tight_layout()
        return fig, {"prediccion": pf, "direccion": dir_txt, "cambio_pct": pct, "ultimo_precio": p0}
    except Exception as e:
        st.caption(f"ℹ️ ARIMA: {e}")
        return None, None


def graficar_garch_forecast(precios, nombre_activo, periodos=10):
    """GARCH(1,1): predicción de volatilidad condicional."""
    if not GARCH_DISPONIBLE:
        return None, None
    try:
        s = precios.dropna().astype(float)
        if len(s) < 100:
            return None, None
        ret = s.pct_change().dropna() * 100
        ret = ret.tail(200)
        fit = arch_model(ret, vol='Garch', p=1, q=1, mean='Zero', rescale=False)\
                  .fit(disp='off', show_warning=False)
        fc  = fit.forecast(horizon=periodos)
        vol_pred = np.sqrt(fc.variance.values[-1, :])
        vol_hist = ret.rolling(5).std().tail(50)
        freq = pd.infer_freq(ret.index[-20:]) or 'h'
        idx_fut = pd.date_range(start=vol_hist.index[-1], periods=periodos + 1, freq=freq)[1:]

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.plot(vol_hist.index, vol_hist.values, color='white', linewidth=1.5, label='Volatilidad histórica')
        ax.plot(idx_fut, vol_pred, color='orange', linewidth=2, linestyle='--', label='Predicción GARCH')
        ax.fill_between(idx_fut, vol_pred * 0.5, vol_pred * 1.5,
                        color='skyblue', alpha=0.35, label='Rango esperado')
        if not np.isnan(vol_hist.values[-1]):
            ax.plot([vol_hist.index[-1], idx_fut[0]],
                    [vol_hist.values[-1], vol_pred[0]], color='orange', linewidth=1.5, linestyle='--')

        vol_media  = float(ret.std())
        vol_futura = float(np.mean(vol_pred))
        ratio = vol_futura / vol_media if vol_media > 0 else 1
        if ratio > 1.5:
            nivel, col = "MUY ALTA ⚠️", "red"
        elif ratio > 1.0:
            nivel, col = "ALTA", "orange"
        elif ratio > 0.5:
            nivel, col = "NORMAL", "lime"
        else:
            nivel, col = "BAJA", "cyan"

        ax.set_title(f"GARCH — {nombre_activo}   Volatilidad: {nivel}", color=col, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', colors='gray', labelsize=7, rotation=45)
        ax.tick_params(axis='y', colors='gray', labelsize=8)
        ax.legend(loc='upper left', fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax.grid(color='gray', linestyle=':', linewidth=0.3, alpha=0.5)
        ax.set_ylabel('Volatilidad (%)', color='gray', fontsize=9)
        plt.tight_layout()
        return fig, {"volatilidad_futura": vol_futura, "nivel": nivel, "vol_media": vol_media}
    except Exception as e:
        st.caption(f"ℹ️ GARCH: {e}")
        return None, None


def graficar_timeframe(df, nombre, tf_label, es_diario=False):
    """
    Gráfico completo por timeframe:
    Precio + POC + VWAP + RSI Divergence + Diamante.
    Retorna (fig, tendencia).
    """
    if df is None or df.empty or len(df) < 15:
        return None, "NEUTRO"

    df = df.copy()
    # Aplanar MultiIndex si existe
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ── POC ──────────────────────────────────────────────────────────────────
    try:
        vol_col = 'Volume'
        bins = 20
        df['_bin'] = pd.cut(df['Close'], bins=bins)
        poc_bin = df.groupby('_bin', observed=True)[vol_col].sum().idxmax()
        poc_price = float((poc_bin.left + poc_bin.right) / 2)
        df.drop(columns=['_bin'], inplace=True)
    except Exception:
        poc_price = float(df['Close'].mean())

    # ── VWAP ─────────────────────────────────────────────────────────────────
    df['_vol'] = df['Volume'].replace(0, 1)
    df['VWAP'] = (df['Close'] * df['_vol']).cumsum() / df['_vol'].cumsum()
    df['VWAP'] = df['VWAP'].ffill().bfill()

    # ── RSI + Divergencias ───────────────────────────────────────────────────
    df['RSI'] = calcular_rsi(df['Close'], periodo=14)
    div_alc, div_baj = detectar_divergencias(df['Close'], df['RSI'], lookback=5)

    # ── Diamante ─────────────────────────────────────────────────────────────
    df['_rvol']  = df['_vol'] / df['_vol'].rolling(20, min_periods=1).mean()
    df['_range'] = df['High'] - df['Low']
    last = df.iloc[-1]
    try:
        rvol_v  = float(last['_rvol'])  if not pd.isna(last['_rvol'])  else 0
        rng_v   = float(last['_range']) if not pd.isna(last['_range']) else 0
        rng_avg = float(df['_range'].rolling(20, min_periods=1).mean().iloc[-1])
        es_diamante = (rvol_v > 2.0) and (rng_v < rng_avg)
    except Exception:
        es_diamante = False

    # ── Tendencia (precio vs VWAP) ───────────────────────────────────────────
    try:
        tendencia  = "COMPRA" if float(last['Close']) > float(last['VWAP']) else "VENTA"
        color_caja = "#155015" if tendencia == "COMPRA" else "#7a1515"
    except Exception:
        tendencia  = "NEUTRO"
        color_caja = "#333333"

    # ── Plot: precio arriba | RSI abajo ──────────────────────────────────────
    tail = 80
    df_p = df.tail(tail)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.2),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)
    fig.patch.set_facecolor('#0e1117')
    ax1.set_facecolor('#0e1117')
    ax2.set_facecolor('#0e1117')

    # Precio
    ax1.plot(df_p.index, df_p['Close'], color='white', alpha=0.85, linewidth=1.3, label='Precio')
    # VWAP
    ax1.plot(df_p.index, df_p['VWAP'], color='cyan', linewidth=1.6, label='VWAP')
    # POC
    ax1.axhline(y=poc_price, color='red', alpha=0.85, linewidth=1.8, label='POC')
    ax1.text(df_p.index[-1], poc_price, f'  {poc_price:,.2f}',
             color='red', fontsize=7.5, fontweight='bold',
             ha='left', va='center', backgroundcolor='#0e1117')

    # Nota delay diario
    if es_diario:
        ax1.text(0.99, 0.03, '⚠️ Último dato: cierre anterior',
                 transform=ax1.transAxes, color='#ffdd57',
                 fontsize=6.5, ha='right', va='bottom', alpha=0.85)

    # Diamante
    if es_diamante:
        ax1.scatter(df_p.index[-1], float(df_p['Close'].iloc[-1]),
                    color='#00d4ff', s=200, marker='d',
                    edgecolors='white', linewidths=1.5, zorder=8, label='💎 Compresión Vol.')

    # Divergencias sobre el precio
    offset = float(df_p['Close'].std()) * 0.45
    n_total = len(df)
    base    = n_total - tail
    for i in div_alc:
        ri = i - base
        if 0 <= ri < len(df_p):
            ax1.scatter(df_p.index[ri],
                        float(df_p['Close'].iloc[ri]) - offset,
                        marker='^', color='lime', s=110, zorder=9,
                        label='Div. Alcista ▲')
    for i in div_baj:
        ri = i - base
        if 0 <= ri < len(df_p):
            ax1.scatter(df_p.index[ri],
                        float(df_p['Close'].iloc[ri]) + offset,
                        marker='v', color='tomato', s=110, zorder=9,
                        label='Div. Bajista ▼')

    # Badge tendencia
    ax1.text(0.02, 0.94, tendencia,
             transform=ax1.transAxes, color='white',
             fontsize=11, fontweight='bold',
             bbox=dict(facecolor=color_caja, alpha=0.92, boxstyle='round,pad=0.4'))

    ax1.set_title(f"TF: {tf_label}  ·  {nombre}  ·  {tendencia}",
                  color='white', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='y', colors='gray', labelsize=7)
    ax1.grid(color='gray', linestyle=':', linewidth=0.3, alpha=0.35)
    # Leyenda sin duplicados
    hdls, lbls = ax1.get_legend_handles_labels()
    ax1.legend(dict(zip(lbls, hdls)).values(), dict(zip(lbls, hdls)).keys(),
               loc='upper right', fontsize=6.5,
               facecolor='#1a1a2e', labelcolor='white')

    # Panel RSI
    rsi_v = df_p['RSI'].values
    ax2.plot(df_p.index, rsi_v, color='#c77dff', linewidth=1.2, label='RSI(14)')
    ax2.axhline(70, color='tomato', linewidth=0.8, linestyle='--', alpha=0.7)
    ax2.axhline(30, color='lime',   linewidth=0.8, linestyle='--', alpha=0.7)
    ax2.axhline(50, color='gray',   linewidth=0.5, linestyle=':', alpha=0.5)
    ax2.fill_between(df_p.index, rsi_v, 70, where=(rsi_v >= 70), color='tomato', alpha=0.18)
    ax2.fill_between(df_p.index, rsi_v, 30, where=(rsi_v <= 30), color='lime',   alpha=0.18)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI', color='gray', fontsize=7)
    ax2.tick_params(axis='x', colors='gray', labelsize=6, rotation=45)
    ax2.tick_params(axis='y', colors='gray', labelsize=6)
    ax2.grid(color='gray', linestyle=':', linewidth=0.3, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=6.5, facecolor='#1a1a2e', labelcolor='white')

    plt.tight_layout(h_pad=0.5)
    return fig, tendencia


# ─── 5. ACTIVOS Y CONFIGURACIÓN ──────────────────────────────────────────────

ACTIVOS = {
    "🥇 Oro (Gold Futures)":        "GC=F",
    "💻 Nasdaq 100 (NQ Futures)":   "NQ=F",
    "₿ Bitcoin (BTC/USD)":          "BTC-USD",
}

# ─── 6. LOOP PRINCIPAL ───────────────────────────────────────────────────────

for nombre, ticker in ACTIVOS.items():
    st.markdown("---")
    st.subheader(nombre)
    consenso = []

    try:
        # ── Descarga base de datos ────────────────────────────────────────────
        df_1h = yf.download(ticker, period="60d", interval="1h", progress=False)
        df_1d = yf.download(ticker, period="1y",  interval="1d", progress=False)

        for _df in [df_1h, df_1d]:
            if isinstance(_df.columns, pd.MultiIndex):
                _df.columns = _df.columns.get_level_values(0)

        df_4h     = resamplear_4h(df_1h)
        df_1h_rec = df_1h.tail(7 * 24)   # ~7 días recientes para el TF de 1H

        # ── ARIMA & GARCH ──────────────────────────────────────────────────────
        st.markdown("##### 🔮 Modelos Cuantitativos — ARIMA · GARCH")
        if not df_1h.empty and len(df_1h) > 100:
            col_a, col_g = st.columns(2)
            with col_a:
                fig_a, res_a = graficar_arima_forecast(df_1h['Close'], nombre)
                if fig_a:
                    st.pyplot(fig_a)
                    plt.close(fig_a)
                    st.caption(
                        f"Precio actual: {res_a['ultimo_precio']:,.2f}  →  "
                        f"Predicción: {res_a['prediccion']:,.2f}  "
                        f"({res_a['cambio_pct']:+.2f}%)"
                    )
                else:
                    st.info("ARIMA no disponible para este activo.")
            with col_g:
                fig_g, res_g = graficar_garch_forecast(df_1h['Close'], nombre)
                if fig_g:
                    st.pyplot(fig_g)
                    plt.close(fig_g)
                    st.caption(
                        f"Volatilidad esperada: {res_g['volatilidad_futura']:.3f}%  |  "
                        f"Media histórica: {res_g['vol_media']:.3f}%  |  "
                        f"Nivel: **{res_g['nivel']}**"
                    )
                else:
                    st.info("GARCH no disponible para este activo.")
        else:
            st.warning("Datos insuficientes para modelos cuantitativos.")

        # ── Análisis Multi-Temporal: 1H · 4H · 1D ────────────────────────────
        st.markdown("##### 📈 Análisis Multi-Temporal — 1H · 4H · Diario")
        col1, col2, col3 = st.columns(3)
        datasets = [
            (col1, df_1h_rec, "1H",  False),
            (col2, df_4h,     "4H",  False),
            (col3, df_1d,     "1D",  True),
        ]
        for col, df_tf, tf_label, es_d in datasets:
            with col:
                fig_tf, tend = graficar_timeframe(df_tf, nombre, tf_label, es_d)
                if fig_tf:
                    st.pyplot(fig_tf)
                    plt.close(fig_tf)
                    consenso.append(tend)
                else:
                    st.warning(f"Sin datos suficientes para {tf_label}")

        # ── Fuego Maestro ──────────────────────────────────────────────────────
        st.markdown("##### 🔥 Consenso Multi-Temporal")
        if len(consenso) >= 2:
            compras = consenso.count("COMPRA")
            ventas  = consenso.count("VENTA")
            tfs_str = " · ".join([
                f"{['1H','4H','1D'][i]} {'✅' if consenso[i]=='COMPRA' else '🔴'}"
                for i in range(len(consenso))
            ])
            if compras == 3:
                st.success(f"🔥 FUEGO MAESTRO — ALINEACIÓN ALCISTA TOTAL  |  {tfs_str}")
                st.caption("Los 3 timeframes confirman precio sobre VWAP. Máxima probabilidad de continuación alcista.")
            elif ventas == 3:
                st.error(f"🧊 ALINEACIÓN BAJISTA TOTAL  |  {tfs_str}")
                st.caption("Los 3 timeframes confirman precio bajo VWAP. Máxima presión vendedora.")
            elif compras == 2:
                st.warning(f"⬆️ SESGO ALCISTA PARCIAL (2/3 TF)  |  {tfs_str}")
                st.caption("Tendencia alcista en desarrollo. Espera que el tercer TF confirme antes de operar.")
            elif ventas == 2:
                st.warning(f"⬇️ SESGO BAJISTA PARCIAL (2/3 TF)  |  {tfs_str}")
                st.caption("Tendencia bajista en desarrollo. Espera confirmación del tercer TF.")
            else:
                st.info(f"⚖️ MERCADO MIXTO  |  {tfs_str}")
                st.caption("Timeframes en conflicto. Sin sesgo claro. No operar hasta alineación.")
        else:
            st.warning("Datos insuficientes para calcular consenso multi-temporal.")

    except Exception as e:
        st.error(f"Error procesando {nombre}: {e}")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "TRIPLE RADAR v10.0  |  ARIMA · GARCH · RSI Divergence · POC (Volume Profile) · VWAP · Compresión de Volatilidad  |  "
    "Datos vía Yahoo Finance  |  No constituye asesoramiento financiero."
)
