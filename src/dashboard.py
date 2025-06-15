import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import joblib
import sys
import os
from pathlib import Path
import st_static_export as sse
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Añadir el directorio src al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modeller import predecir
from src.logger import LoggerService

def calcular_kpis(df):
    """Calcula los KPIs principales para el dashboard"""
    # Asegurarse que la fecha está en formato datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Tasa de variación diaria
    df['tasa_variacion'] = df['close'].pct_change() * 100
    
    # 2. Media móvil de 20 días
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # 3. Volatilidad (desviación estándar de 20 días)
    df['volatilidad'] = df['tasa_variacion'].rolling(window=20).std()
    
    # 4. Retorno acumulado
    df['retorno_acumulado'] = (1 + df['tasa_variacion']/100).cumprod() - 1
    
    # 5. Desviación estándar de 20 días
    df['desviacion_std'] = df['close'].rolling(window=20).std()
    
    return df

def crear_grafico_precio_volumen(df):
    """Crea un gráfico de precio y volumen usando plotly"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       row_heights=[0.7, 0.3])

    # Gráfico de precios
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Precio'
        ),
        row=1, col=1
    )

    # Media móvil
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['sma_20'],
            name='SMA 20',
            line=dict(color='orange')
        ),
        row=1, col=1
    )

    # Volumen
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='Volumen'
        ),
        row=2, col=1
    )

    fig.update_layout(
        title='Precio y Volumen',
        yaxis_title='Precio',
        yaxis2_title='Volumen',
        xaxis_rangeslider_visible=False
    )

    return fig

def crear_grafico_precio_altair(df):
    """Crea un gráfico de precio usando Altair"""
    try:
        # Asegurarse que los datos están limpios y formateados
        df_clean = df.copy()
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        df_clean = df_clean.dropna(subset=['close', 'volume', 'sma_20'])
        
        # Preparar datos para el volumen
        df_clean['color'] = np.where(df_clean['close'] > df_clean['close'].shift(1), 'green', 'red')
        
        # Gráfico de precios
        base = alt.Chart(df_clean).encode(
            x=alt.X('date:T', title='Fecha'),
            y=alt.Y('close:Q', title='Precio'),
            tooltip=['date', 'close', 'volume']
        ).properties(
            width='container'
        )
        
        # Línea de precios
        line = base.mark_line(color='blue').encode(
            y='close:Q'
        )
        
        # Media móvil
        sma = base.mark_line(color='orange').encode(
            y='sma_20:Q'
        )
        
        # Volumen
        volume = alt.Chart(df_clean).mark_bar().encode(
            x=alt.X('date:T', title='Fecha'),
            y=alt.Y('volume:Q', title='Volumen'),
            color=alt.Color('color:N', scale=alt.Scale(domain=['green', 'red'], range=['green', 'red']))
        ).properties(
            height=100,
            width='container'
        )
        
        # Combinar gráficos
        chart = alt.vconcat(
            (line + sma).properties(
                height=300,
                title='Precio y SMA 20',
                width='container'
            ),
            volume.properties(
                title='Volumen',
                width='container'
            )
        ).resolve_scale(x='shared')
        
        return chart
    except Exception as e:
        print(f"Error al crear gráfico de precio: {str(e)}")
        return None

def crear_grafico_retornos_altair(df):
    """Crea un gráfico de distribución de retornos usando Altair"""
    try:
        # Asegurarse que los datos están limpios
        df_clean = df.copy()
        df_clean = df_clean.dropna(subset=['tasa_variacion'])
        
        return alt.Chart(df_clean).mark_bar().encode(
            alt.X('tasa_variacion:Q', 
                  bin=alt.Bin(maxbins=50), 
                  title='Retorno Diario'),
            alt.Y('count()', title='Frecuencia'),
            tooltip=['count()']
        ).properties(
            title='Distribución de Retornos Diarios',
            height=300,
            width='container'
        )
    except Exception as e:
        print(f"Error al crear gráfico de retornos: {str(e)}")
        return None
    
def crear_grafico_arima(df):
    """Crea un gráfico de ARIMA usando Altair"""
    try:
        # Asegurarse que los datos están limpios
        train_size = int(len(df) * 0.8)
        train = df.iloc[:train_size]
        test = df.iloc[train_size:]

        # Cargar modelo ARIMA
        with open("src/static/models/modelo_arima.pkl", "rb") as f:
            modelo_arima = pickle.load(f)

        # Obtener pronóstico
        forecast = modelo_arima.forecast(steps=len(test))

        # Reconstruir los valores si el modelo fue entrenado con diferencias
        last_value = train["close"].iloc[-1]
        forecast_values = forecast.cumsum() + last_value
        forecast_values.index = test.index

        # Métricas
        st.write("Métricas del modelo ARIMA")
        rmse = np.sqrt(mean_squared_error(test["close"], forecast_values))
        mae = mean_absolute_error(test["close"], forecast_values)
        mape = np.mean(np.abs((test["close"] - forecast_values) / test["close"])) * 100
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"RMSE: {rmse:.2f}")
        with col2:
            st.write(f"MAE: {mae:.2f}")
        with col3:
            st.write(f"MAPE: {mape:.2f}%")

        st.write("Gráfico del pronóstico")
        # Gráfico
        plt.figure(figsize=(12, 5))
        plt.plot(train["close"], label="Entrenamiento")
        plt.plot(test["close"], label="Real (Test)", color="gray")
        plt.plot(forecast_values, label="Pronóstico ARIMA", linestyle="--", color="orange")
        plt.title("Pronóstico con modelo ARIMA cargado")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)

    except Exception as e:
        print(f"Error al crear gráfico de ARIMA: {str(e)}")
        return None
    
def data_arima(df):
    """Crea un gráfico de ARIMA usando Altair"""
    try:
        # Asegurarse que los datos están limpios
        train_size = int(len(df) * 0.8)
        train = df.iloc[:train_size]
        test = df.iloc[train_size:]

        # Cargar modelo ARIMA
        with open("src/static/models/modelo_arima.pkl", "rb") as f:
            modelo_arima = pickle.load(f)

        # Obtener pronóstico
        forecast = modelo_arima.forecast(steps=len(test))

        # Reconstruir los valores si el modelo fue entrenado con diferencias
        last_value = train["close"].iloc[-1]
        forecast_values = forecast.cumsum() + last_value
        forecast_values.index = test.index

        # Métricas
        st.write("Métricas del modelo ARIMA")
        rmse = np.sqrt(mean_squared_error(test["close"], forecast_values))
        mae = mean_absolute_error(test["close"], forecast_values)
        mape = np.mean(np.abs((test["close"] - forecast_values) / test["close"])) * 100

        return f"RMSE: {rmse:.2f} \n MAE: {mae:.2f} \n MAPE: {mape:.2f}%"



    
    except Exception as e:
        print(f"Error al crear datos de ARIMA: {str(e)}")
        return None

def crear_grafico_arima_altair(df):
    """Crea un gráfico de ARIMA usando Altair"""
    try:
        # Asegurarse que los datos están limpios
        train_size = int(len(df) * 0.8)
        train = df.iloc[:train_size].copy()
        test = df.iloc[train_size:].copy()

        # Cargar modelo ARIMA
        with open("src/static/models/modelo_arima.pkl", "rb") as f:
            modelo_arima = pickle.load(f)

        # Obtener pronóstico
        forecast = modelo_arima.forecast(steps=len(test))

        # Reconstruir los valores si el modelo fue entrenado con diferencias
        last_value = train["close"].iloc[-1]
        forecast_values = forecast.cumsum() + last_value
        forecast_values.index = test.index

        # Preparar datos para Altair
        # Datos de entrenamiento
        train_data = train.reset_index()
        train_data['tipo'] = 'Entrenamiento'
        train_data = train_data[['date', 'close', 'tipo']]
        
        # Datos de test real
        test_data = test.reset_index()
        test_data['tipo'] = 'Real (Test)'
        test_data = test_data[['date', 'close', 'tipo']]
        
        # Datos de pronóstico
        forecast_data = pd.DataFrame({
            'date': test['date'],
            'close': forecast_values,
            'tipo': 'Pronóstico ARIMA'
        })
        
        # Combinar todos los datos
        all_data = pd.concat([train_data, test_data, forecast_data], ignore_index=True)
        
        # Crear gráfico con Altair
        chart = alt.Chart(all_data).mark_line().encode(
            x=alt.X('date:T', title='Fecha'),
            y=alt.Y('close:Q', title='Precio de Cierre'),
            color=alt.Color('tipo:N', 
                          scale=alt.Scale(
                              domain=['Entrenamiento', 'Real (Test)', 'Pronóstico ARIMA'],
                              range=['blue', 'gray', 'orange']
                          )),
            strokeDash=alt.StrokeDash('tipo:N',
                                    scale=alt.Scale(
                                        domain=['Entrenamiento', 'Real (Test)', 'Pronóstico ARIMA'],
                                        range=[0, 0, 5]  # Línea punteada solo para pronóstico
                                    )),
            tooltip=['date', 'close', 'tipo']
        ).properties(
            title='Pronóstico con modelo ARIMA',
            width='container',
            height=400
        ).configure_axis(
            grid=True
        ).configure_legend(
            orient='top'
        )
        
        return chart

    except Exception as e:
        print(f"Error al crear gráfico de ARIMA: {str(e)}")
        return None

def crear_grafico_volatilidad_altair(df):
    """Crea un gráfico de volatilidad vs retorno usando Altair"""
    try:
        # Asegurarse que los datos están limpios
        df_clean = df.copy()
        df_clean = df_clean.dropna(subset=['volatilidad', 'tasa_variacion'])
        
        return alt.Chart(df_clean).mark_circle().encode(
            x=alt.X('volatilidad:Q', title='Volatilidad'),
            y=alt.Y('tasa_variacion:Q', title='Retorno Diario'),
            tooltip=['date', 'volatilidad', 'tasa_variacion']
        ).properties(
            title='Volatilidad vs Retorno Diario',
            height=300,
            width='container'
        )
    except Exception as e:
        print(f"Error al crear gráfico de volatilidad: {str(e)}")
        return None

def mostrar_dashboard():
    """Función principal que muestra el dashboard"""
    st.set_page_config(page_title="Dashboard de Trading", layout="wide")
    
    # Título
    st.title("Dashboard de Análisis de Trading")
    
    try:
        # Cargar datos
        df = pd.read_csv('src/static/data/historical.csv')
        df = calcular_kpis(df)
        
        # Cargar modelo y métricas
        model_data = joblib.load('src/static/models/model.pkl')
        metrics = model_data.get('metrics', {})
        
        # Obtener última predicción
        logger = LoggerService()
        ultima_prediccion = predecir(logger)
        
        # Layout de 3 columnas para KPIs principales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Última Predicción",
                "Subida" if ultima_prediccion['prediccion'] == 1 else "Bajada",
                f"Probabilidad: {ultima_prediccion['probabilidad']:.2%}"
            )
            
        with col2:
            st.metric(
                "Retorno Acumulado",
                f"{df['retorno_acumulado'].iloc[-1]:.2%}",
                f"Variación diaria: {df['tasa_variacion'].iloc[-1]:.2%}"
            )
            
        with col3:
            st.metric(
                "Volatilidad (20d)",
                f"{df['volatilidad'].iloc[-1]:.2%}",
                f"Desviación Std: {df['desviacion_std'].iloc[-1]:.2f}"
            )
        
        # Gráfico principal
        st.plotly_chart(crear_grafico_precio_volumen(df), use_container_width=True)
        
        # Métricas del modelo
        st.subheader("Métricas del Modelo")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        with col2:
            st.metric("F1-Score", f"{metrics.get('f1', 0):.2%}")
        with col3:
            st.metric("AUC", f"{metrics.get('auc', 0):.2%}")
        with col4:
            st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
        with col5:
            st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
        
        # Gráficos adicionales
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribución de Retornos")
            fig_retornos = go.Figure()
            fig_retornos.add_trace(go.Histogram(x=df['tasa_variacion'], nbinsx=50))
            fig_retornos.update_layout(title="Distribución de Retornos Diarios")
            st.plotly_chart(fig_retornos, use_container_width=True)
            
        with col2:
            st.subheader("Volatilidad vs Retorno")
            fig_vol_ret = go.Figure()
            fig_vol_ret.add_trace(go.Scatter(
                x=df['volatilidad'],
                y=df['tasa_variacion'],
                mode='markers',
                name='Volatilidad vs Retorno'
            ))
            fig_vol_ret.update_layout(
                title="Volatilidad vs Retorno Diario",
                xaxis_title="Volatilidad",
                yaxis_title="Retorno Diario"
            )
            st.plotly_chart(fig_vol_ret, use_container_width=True)
        
        # Tabla de datos recientes
        st.subheader("Datos Recientes")
        st.dataframe(
            df.tail(10)[['date', 'close', 'tasa_variacion', 'sma_20', 'volatilidad', 'retorno_acumulado']]
            .style.format({
                'tasa_variacion': '{:.2%}',
                'sma_20': '{:.2f}',
                'volatilidad': '{:.2%}',
                'retorno_acumulado': '{:.2%}'
            })
        )

        # Gráfico de ARIMA
        st.subheader("Pronóstico de Precio de Cierre")
        crear_grafico_arima(df)
        
    except Exception as e:
        st.error(f"Error al cargar el dashboard: {str(e)}")

def guardar_dashboard_como_html():
    """
    Guarda el dashboard como un archivo HTML estático usando st-static-export.
    """
    try:
        # Crear directorio para el HTML si no existe
        output_dir = Path('src/static/dashboard')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Nombre del archivo con timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'dashboard_{timestamp}.html'
        
        # CSS personalizado para mejorar la apariencia
        css_text = """
        body {
            max-width: 100%;
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            box-sizing: border-box;
        }
        * {
            box-sizing: border-box;
        }
        div {
            width: 100%;
            max-width: 100%;
        }
        table, th, td {
            border: 1px solid black;
            border-collapse: collapse;
            width: 100%;
        }
        tr:nth-child(even) {background-color: #f2f2f2;}
        .table {
            width: 100%;
        }
        .metric {
            padding: 1rem;
            margin: 0.5rem;
            border-radius: 0.5rem;
            background-color: #f8f9fa;
            width: 100%;
        }
        .kpi-container {
            display: flex;
            justify-content: space-between;
            margin: 1rem 0;
            width: 100%;
        }
        .kpi-box {
            flex: 1;
            margin: 0 0.5rem;
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 0.5rem;
            text-align: center;
        }
        .footnote {
            color: #666;
            font-size: 0.8rem;
            margin-top: 2rem;
            text-align: center;
        }
        .chart-container {
            width: 100%;
            margin: 2rem 0;
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 0.5rem;
        }
        .vega-embed {
            width: 100% !important;
            max-width: 100% !important;
        }
        .vega-embed canvas {
            width: 100% !important;
            height: auto !important;
        }
        .stApp {
            max-width: 100% !important;
            padding: 0 !important;
        }
        """
        
        # Inicializar el exportador estático
        static_html = sse.StreamlitStaticExport(css=css_text)
        
        # Cargar datos y preparar componentes
        df = pd.read_csv('src/static/data/historical.csv')
        df = calcular_kpis(df)
        
        # Verificar que los datos están correctos
        if df.empty:
            raise ValueError("El DataFrame está vacío después de cargar los datos")
            
        model_data = joblib.load('src/static/models/model.pkl')
        metrics = model_data.get('metrics', {})
        logger = LoggerService()
        ultima_prediccion = predecir(logger)
        
        # 1. Título principal
        static_html.add_header(id="dashboard_title", text="Dashboard de Análisis de Trading", size="H1")
        
        # 2. KPIs Principales
        static_html.add_header(id="kpis_section", text="KPIs Principales", size="H2")
        kpis_text = f"""
        Última Predicción: {'Subida' if ultima_prediccion['prediccion'] == 1 else 'Bajada'} (Probabilidad: {ultima_prediccion['probabilidad']:.2%})
        Retorno Acumulado: {df['retorno_acumulado'].iloc[-1]:.2%} (Variación diaria: {df['tasa_variacion'].iloc[-1]:.2%})
        Volatilidad (20d): {df['volatilidad'].iloc[-1]:.2%} (Desviación Std: {df['desviacion_std'].iloc[-1]:.2f})
        """
        static_html.add_text(id="kpis_text", text=kpis_text, text_class="metric")
        
        # 3. Gráfico Principal
        static_html.add_header(id="main_chart", text="Gráfico de Precio y Volumen", size="H2")
        chart_precio = crear_grafico_precio_altair(df)
        if chart_precio is not None:
            static_html.export_altair_graph(id="precio_chart", graph=chart_precio)
        else:
            static_html.add_text(id="precio_error", text="No se pudo generar el gráfico de precio", text_class="metric")
        
        # 4. Métricas del Modelo
        static_html.add_header(id="model_metrics", text="Métricas del Modelo", size="H2")
        metrics_text = f"""
        Accuracy: {metrics.get('accuracy', 0):.2%}
        F1-Score: {metrics.get('f1', 0):.2%}
        AUC: {metrics.get('auc', 0):.2%}
        RMSE: {metrics.get('rmse', 0):.4f}
        MAE: {metrics.get('mae', 0):.4f}
        """
        static_html.add_text(id="metrics_text", text=metrics_text, text_class="metric")
        
        # 5. Gráficos Adicionales
        static_html.add_header(id="additional_charts", text="Análisis Adicional", size="H2")
        
        # Gráfico de distribución de retornos
        chart_retornos = crear_grafico_retornos_altair(df)
        if chart_retornos is not None:
            static_html.export_altair_graph(id="retornos_chart", graph=chart_retornos)
        else:
            static_html.add_text(id="retornos_error", text="No se pudo generar el gráfico de retornos", text_class="metric")
        
        # Gráfico de volatilidad vs retorno
        chart_volatilidad = crear_grafico_volatilidad_altair(df)
        if chart_volatilidad is not None:
            static_html.export_altair_graph(id="volatilidad_chart", graph=chart_volatilidad)
        else:
            static_html.add_text(id="volatilidad_error", text="No se pudo generar el gráfico de volatilidad", text_class="metric")
        
        # 6. Datos Recientes
        static_html.add_header(id="recent_data", text="Datos Recientes", size="H2")
        df_recent = df.tail(10)[['date', 'close', 'tasa_variacion', 'sma_20', 'volatilidad', 'retorno_acumulado']]
        static_html.export_dataframe(
            id="recent_data_table",
            dataframe=df_recent,
            table_class="table",
            inside_expandable=True
        )
        
        # 7. Nota al pie
        static_html.add_text(
            id="footnote",
            text="Dashboard generado automáticamente con st-static-export y Streamlit",
            text_class="footnote"
        )

        # 8. Datos de ARIMA
        static_html.add_header(id="arima_data", text="Datos de ARIMA", size="H2")
        arima_data = data_arima(df)
        static_html.add_text(id="arima_data_text", text=arima_data, text_class="metric")

        # 8. Gráfico de ARIMA
        static_html.add_header(id="arima_chart", text="Pronóstico de Precio de Cierre", size="H2")
        chart_arima = crear_grafico_arima_altair(df)
        if chart_arima is not None:
            static_html.export_altair_graph(id="arima_chart", graph=chart_arima)
        else:
            static_html.add_text(id="arima_error", text="No se pudo generar el gráfico de ARIMA", text_class="metric")
        
        # Crear y guardar el HTML
        str_result = static_html.create_html(return_type="String")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(str_result)
        
        print(f"Dashboard guardado como HTML estático en: {output_file}")
        return str(output_file)
        
    except Exception as e:
        print(f"Error al guardar el dashboard: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--save-html':
        guardar_dashboard_como_html()
    else:
        mostrar_dashboard()
