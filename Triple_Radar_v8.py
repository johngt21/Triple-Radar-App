import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Configuracion de pagina - DEBE IR PRIMERO
st.set_page_config(page_title="TRIPLE RADAR v9.0", layout="wide", page_icon="🔥")

# --- 1. SEGURIDAD ---
def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("TRIPLE RADAR v9.0 - Acceso Restringido", type="password", on_change=password_entered, key="password")
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

st.title("🎛️ TRIPLE RADAR v9.0: ARIMA + GARCH + Analisis Tecnico")

# --- 2. LEYENDA UNICA EXPLICATIVA (PARA DUMMIES) ---
with st.expander("📖 GUIA RAPIDA: Que significa todo esto? (Lee esto primero)", expanded=True):
    st.markdown("""
### 🎯 Este sistema te ayuda a decidir si COMPRAR, VENDER o ESPERAR

| Elemento | Que es | Como usarlo |
|----------|--------|-------------|
| **POC (Linea Roja)** | El precio donde hubo MAS compradores y vendedores. | Si el precio esta cerca, puede rebotar o romper con fuerza. |
| **VWAP (Linea Cyan)** | Precio promedio ponderado por volumen del dia. | Precio ARRIBA = alcista. Precio ABAJO = bajista. |
| **Diamante Azul** | Senal de posible reversion. Hay mucho volumen pero poca volatilidad. | El precio podria cambiar de direccion pronto. |
| **Grafico ARIMA** | Linea de prediccion futura con bandas azul cielo (95% confianza). | Si la linea sube = tendencia alcista. Si baja = bajista. |
| **Grafico GARCH** | Muestra la volatilidad esperada con bandas de incertidumbre. | Bandas anchas = mas riesgo. Bandas estrechas = menos riesgo. |
| **FUEGO MAESTRO** | Senal cuando 5m, 15m y 1H coinciden en la misma direccion. | Esta es la senal mas fuerte! |

---

### ⚡ REGLAS SIMPLES PARA OPERAR:

1. **Si ves FUEGO MAESTRO** -> Considera entrar en esa direccion
2. **Si ARIMA apunta ARRIBA + bandas estrechas** -> Buena oportunidad de COMPRA
3. **Si ARIMA apunta ABAJO + bandas estrechas** -> Buena oportunidad de VENTA  
4. **Si las bandas GARCH son MUY ANCHAS** -> CUIDADO! Reduce tu riesgo
5. **Si los timeframes no coinciden** -> NO OPERES, espera alineacion

*Recuerda: Ningun sistema es 100% efectivo. Siempre usa stop loss.*
""")

# --- 3. FUNCIONES DE ARIMA Y GARCH ---

# Importar ARIMA y GARCH con manejo de errores
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_DISPONIBLE = True
except ImportError:
    ARIMA_DISPONIBLE = False
    st.warning("statsmodels no instalado. ARIMA no disponible. Instala con: pip install statsmodels")

try:
    from arch import arch_model
    GARCH_DISPONIBLE = True
except ImportError:
    GARCH_DISPONIBLE = False
    st.warning("arch no instalado. GARCH no disponible. Instala con: pip install arch")


def graficar_arima_forecast(precios, nombre_activo, periodos_prediccion=10):
    """
    ARIMA: Grafica prediccion de precios con bandas de confianza 95%
    """
    if not ARIMA_DISPONIBLE:
        return None, None
    
    try:
        precios_clean = precios.dropna().astype(float)
        if len(precios_clean) < 50:
            return None, None
        
        # Usar ultimos 200 datos para mejor rendimiento
        precios_clean = precios_clean.tail(200)
        
        # Ajustar modelo ARIMA
        model = ARIMA(precios_clean, order=(2, 1, 2))
        model_fit = model.fit()
        
        # Forecast con intervalos de confianza
        forecast_result = model_fit.get_forecast(steps=periodos_prediccion)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)  # 95% confianza
        
        # Crear indice futuro
        ultimo_indice = precios_clean.index[-1]
        if isinstance(ultimo_indice, pd.Timestamp):
            freq = pd.infer_freq(precios_clean.index[-20:])
            if freq is None:
                freq = 'H'  # Default a horas
            indice_futuro = pd.date_range(start=ultimo_indice, periods=periodos_prediccion + 1, freq=freq)[1:]
        else:
            indice_futuro = range(len(precios_clean), len(precios_clean) + periodos_prediccion)
        
        # Crear grafico
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        
        # Ultimos 50 precios historicos
        historico = precios_clean.tail(50)
        ax.plot(historico.index, historico.values, color='white', linewidth=1.5, label='Precio Real')
        
        # Linea de forecast
        ax.plot(indice_futuro, forecast_mean.values, color='lime', linewidth=2, label='Prediccion ARIMA', linestyle='--')
        
        # Bandas de confianza 95% en skyblue
        ax.fill_between(indice_futuro, 
                        conf_int.iloc[:, 0].values, 
                        conf_int.iloc[:, 1].values, 
                        color='skyblue', alpha=0.4, label='Intervalo 95%')
        
        # Linea conectora
        ax.plot([historico.index[-1], indice_futuro[0]], 
                [historico.values[-1], forecast_mean.values[0]], 
                color='lime', linewidth=1.5, linestyle='--')
        
        # Calcular direccion
        ultimo_precio = float(precios_clean.iloc[-1])
        precio_predicho = float(forecast_mean.iloc[-1])
        direccion = "SUBE" if precio_predicho > ultimo_precio else "BAJA"
        cambio_pct = ((precio_predicho - ultimo_precio) / ultimo_precio) * 100
        
        ax.set_title(f"ARIMA Forecast - {nombre_activo} | {direccion} {cambio_pct:+.2f}%", color='white', fontsize=11)
        ax.tick_params(axis='x', colors='gray', labelsize=7, rotation=45)
        ax.tick_params(axis='y', colors='gray', labelsize=8)
        ax.legend(loc='upper left', fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax.grid(color='gray', linestyle=':', linewidth=0.3, alpha=0.5)
        plt.tight_layout()
        
        resultado = {
            "prediccion": precio_predicho,
            "direccion": direccion,
            "cambio_pct": cambio_pct,
            "ultimo_precio": ultimo_precio
        }
        
        return fig, resultado
        
    except Exception as e:
        st.warning(f"Error en ARIMA: {e}")
        return None, None


def graficar_garch_forecast(precios, nombre_activo, periodos_prediccion=10):
    """
    GARCH: Grafica volatilidad predicha con bandas de confianza
    """
    if not GARCH_DISPONIBLE:
        return None, None
    
    try:
        precios_clean = precios.dropna().astype(float)
        if len(precios_clean) < 100:
            return None, None
        
        # Calcular retornos porcentuales
        retornos = precios_clean.pct_change().dropna() * 100
        retornos = retornos.tail(200)
        
        # Ajustar modelo GARCH(1,1)
        model = arch_model(retornos, vol='Garch', p=1, q=1, mean='Zero', rescale=False)
        model_fit = model.fit(disp='off', show_warning=False)
        
        # Forecast de varianza
        forecast = model_fit.forecast(horizon=periodos_prediccion)
        varianza_predicha = forecast.variance.values[-1, :]
        volatilidad_predicha = np.sqrt(varianza_predicha)
        
        # Volatilidad historica (ultimos 50 periodos)
        vol_historica = retornos.rolling(window=5).std().tail(50)
        
        # Crear indice futuro
        ultimo_indice = vol_historica.index[-1]
        if isinstance(ultimo_indice, pd.Timestamp):
            freq = pd.infer_freq(retornos.index[-20:])
            if freq is None:
                freq = 'H'
            indice_futuro = pd.date_range(start=ultimo_indice, periods=periodos_prediccion + 1, freq=freq)[1:]
        else:
            indice_futuro = range(len(vol_historica), len(vol_historica) + periodos_prediccion)
        
        # Crear grafico
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        
        # Volatilidad historica
        ax.plot(vol_historica.index, vol_historica.values, color='white', linewidth=1.5, label='Volatilidad Real')
        
        # Linea de forecast volatilidad
        ax.plot(indice_futuro, volatilidad_predicha, color='orange', linewidth=2, label='Prediccion GARCH', linestyle='--')
        
        # Bandas de confianza (aproximacion: +/- 1.96 * volatilidad)
        banda_superior = volatilidad_predicha * 1.5
        banda_inferior = volatilidad_predicha * 0.5
        ax.fill_between(indice_futuro, 
                        banda_inferior, 
                        banda_superior, 
                        color='skyblue', alpha=0.4, label='Rango Esperado')
        
        # Linea conectora
        if len(vol_historica) > 0 and not np.isnan(vol_historica.values[-1]):
            ax.plot([vol_historica.index[-1], indice_futuro[0]], 
                    [vol_historica.values[-1], volatilidad_predicha[0]], 
                    color='orange', linewidth=1.5, linestyle='--')
        
        # Determinar nivel de volatilidad
        vol_promedio = float(retornos.std())
        vol_futura = float(np.mean(volatilidad_predicha))
        
        if vol_futura > vol_promedio * 1.5:
            nivel = "MUY ALTA"
            nivel_color = "red"
        elif vol_futura > vol_promedio:
            nivel = "ALTA"
            nivel_color = "orange"
        elif vol_futura > vol_promedio * 0.5:
            nivel = "NORMAL"
            nivel_color = "lime"
        else:
            nivel = "BAJA"
            nivel_color = "cyan"
        
        ax.set_title(f"GARCH Volatilidad - {nombre_activo} | Nivel: {nivel}", color=nivel_color, fontsize=11)
        ax.tick_params(axis='x', colors='gray', labelsize=7, rotation=45)
        ax.tick_params(axis='y', colors='gray', labelsize=8)
        ax.legend(loc='upper left', fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax.grid(color='gray', linestyle=':', linewidth=0.3, alpha=0.5)
        ax.set_ylabel('Volatilidad (%)', color='gray', fontsize=9)
        plt.tight_layout()
        
        resultado = {
            "volatilidad_futura": vol_futura,
            "nivel": nivel,
            "volatilidad_actual": vol_promedio
        }
        
        return fig, resultado
        
    except Exception as e:
        st.warning(f"Error en GARCH: {e}")
        return None, None


# --- 4. ANALISIS DE MERCADO ---
activos = {
    "Oro (Gold)": "GC=F", 
    "Yen (USD/JPY)": "USDJPY=X", 
    "Bitcoin (BTC/USD)": "BTC-USD"
}

tfs = {"5m": "2d", "15m": "5d", "1h": "30d"}

for nombre, ticker in activos.items():
    st.markdown("---")
    st.subheader(f"📊 {nombre}")
    
    consenso_tendencia = [] 

    try:
        # --- SECCION ARIMA Y GARCH CON GRAFICOS ---
        st.markdown("##### 🔮 Predicciones ARIMA & GARCH")
        
        # Descargar datos para analisis (1 hora, 60 dias)
        df_analisis = yf.download(ticker, period="60d", interval="1h", progress=False)
        if isinstance(df_analisis.columns, pd.MultiIndex):
            df_analisis.columns = df_analisis.columns.get_level_values(0)
        
        if not df_analisis.empty and len(df_analisis) > 100:
            col_arima, col_garch = st.columns(2)
            
            with col_arima:
                fig_arima, res_arima = graficar_arima_forecast(df_analisis['Close'], nombre)
                if fig_arima:
                    st.pyplot(fig_arima)
                    plt.close(fig_arima)
                    st.caption(f"Precio actual: ${res_arima['ultimo_precio']:.2f} → Prediccion: ${res_arima['prediccion']:.2f}")
                else:
                    st.info("ARIMA: Datos insuficientes o error en calculo")
            
            with col_garch:
                fig_garch, res_garch = graficar_garch_forecast(df_analisis['Close'], nombre)
                if fig_garch:
                    st.pyplot(fig_garch)
                    plt.close(fig_garch)
                    st.caption(f"Volatilidad esperada: {res_garch['volatilidad_futura']:.2f}% | Nivel: {res_garch['nivel']}")
                else:
                    st.info("GARCH: Datos insuficientes o error en calculo")
        else:
            st.warning("Datos insuficientes para ARIMA/GARCH")

        # --- GRAFICOS POR TIMEFRAME ---
        st.markdown("##### 📈 Analisis por Timeframe")
        cols = st.columns(3)
        
        for idx, (tf, per) in enumerate(tfs.items()):
            df = yf.download(ticker, period=per, interval=tf, progress=False)
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.get_level_values(0)
            
            if df.empty or len(df) < 10:
                with cols[idx]:
                    st.warning(f"Sin datos para {tf}")
                continue

            # POC (Point of Control)
            try:
                bins = 20
                df['price_bin'] = pd.cut(df['Close'], bins=bins)
                vol_by_bin = df.groupby('price_bin', observed=True)['Volume'].sum()
                poc_idx = vol_by_bin.idxmax()
                poc_price = float((poc_idx.left + poc_idx.right) / 2)
            except:
                poc_price = float(df['Close'].mean())
            
            # VWAP - Calcular correctamente para todos los activos
            # Para divisas, el volumen puede ser bajo o cero, usamos tick count como proxy
            df['Volume_adj'] = df['Volume'].replace(0, 1)  # Evitar division por cero
            if df['Volume_adj'].sum() > 0:
                df['VWAP'] = (df['Close'] * df['Volume_adj']).cumsum() / df['Volume_adj'].cumsum()
            else:
                # Si no hay volumen, usar SMA como proxy
                df['VWAP'] = df['Close'].rolling(window=20, min_periods=1).mean()
            
            # Asegurarse de que VWAP no tenga NaN
            df['VWAP'] = df['VWAP'].ffill().bfill()
            
            # Diamante (alta volumen, baja volatilidad)
            df['RVOL'] = df['Volume_adj'] / df['Volume_adj'].rolling(20, min_periods=1).mean()
            df['Range'] = df['High'] - df['Low']
            last = df.iloc[-1]
            
            try:
                rvol_val = float(last['RVOL']) if not pd.isna(last['RVOL']) else 0
                range_val = float(last['Range']) if not pd.isna(last['Range']) else 0
                range_mean = float(df['Range'].rolling(20, min_periods=1).mean().iloc[-1])
                es_diamante = (rvol_val > 2.0) and (range_val < range_mean)
            except:
                es_diamante = False

            # Determinar tendencia
            tendencia = "NEUTRO"
            color_box = "gray"
            try:
                close_val = float(last['Close'])
                vwap_val = float(last['VWAP'])
                if close_val > vwap_val:
                    tendencia = "COMPRA"
                    color_box = "green"
                    consenso_tendencia.append("COMPRA")
                else:
                    tendencia = "VENTA"
                    color_box = "red"
                    consenso_tendencia.append("VENTA")
            except:
                consenso_tendencia.append("NEUTRO")

            # Graficar
            with cols[idx]:
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')
                
                # Precio (linea blanca)
                ax.plot(df.index, df['Close'], color='white', alpha=0.7, linewidth=1.2, label='Precio')
                
                # VWAP (linea cyan) - ASEGURAR QUE SE MUESTRE
                ax.plot(df.index, df['VWAP'], color='cyan', linestyle='-', alpha=0.9, linewidth=1.5, label='VWAP')
                
                # POC (linea roja horizontal)
                ax.axhline(y=poc_price, color='red', alpha=0.8, linewidth=2, label='POC')
                ax.text(df.index[-1], poc_price, f' {poc_price:.2f}', 
                        color='red', fontsize=9, fontweight='bold', 
                        ha='left', va='center', backgroundcolor='#0e1117')

                # Cuadro de tendencia
                ax.text(0.05, 0.92, tendencia, transform=ax.transAxes, 
                        color='white', fontsize=11, fontweight='bold', 
                        bbox=dict(facecolor=color_box, alpha=0.8, boxstyle='round,pad=0.5'))

                # Diamante si aplica
                if es_diamante:
                    ax.scatter(df.index[-1], float(df['Close'].iloc[-1]), 
                              color='#00d4ff', s=150, marker='d', 
                              edgecolors='white', linewidths=2, zorder=5, label='Diamante')

                ax.set_title(f"TF: {tf} | {tendencia}", color="white", fontsize=11, fontweight='bold')
                ax.tick_params(axis='x', colors='gray', labelsize=6, rotation=45)
                ax.tick_params(axis='y', colors='gray', labelsize=7)
                ax.grid(color='gray', linestyle=':', linewidth=0.3, alpha=0.4)
                ax.legend(loc='upper right', fontsize=7, facecolor='#1a1a2e', labelcolor='white')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # --- CONCLUSION FUEGO MAESTRO ---
        st.markdown("##### 🔮 Conclusion del Algoritmo:")
        
        if len(consenso_tendencia) == 3:
            if all(t == "COMPRA" for t in consenso_tendencia):
                st.success("🔥🔥🔥 FUEGO MAESTRO DETECTADO! ALINEACION TOTAL DE COMPRA 🔥🔥🔥")
                st.caption(f"Fuerte presion de compra en {nombre} (5m, 15m, 1h).")
            elif all(t == "VENTA" for t in consenso_tendencia):
                st.error("🧊🧊🧊 VENTA FUERTE CONFIRMADA! ALINEACION TOTAL BAJISTA 🧊🧊🧊")
                st.caption(f"Fuerte presion de venta en {nombre} (5m, 15m, 1h).")
            else:
                st.info("⚖️ MERCADO MIXTO: Los tiempos no coinciden. Espera mejor alineacion.")
        else:
            st.warning("Datos insuficientes para calculo maestro.")

    except Exception as e: 
        st.error(f"Error procesando {nombre}: {e}")

st.markdown("---")
st.caption("TRIPLE RADAR v9.0 | ARIMA + GARCH + Analisis Cuantitativo")
