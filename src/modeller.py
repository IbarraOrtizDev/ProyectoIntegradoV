import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from xgboost import XGBClassifier
import joblib
import os
from enricher import enriquecer_datos, obtener_features
from logger import LoggerService
from service.edaService import realizar_eda
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import pickle

def entrenar(logger: LoggerService):
    """
    Entrena el modelo y lo guarda como model.pkl junto con el threshold óptimo
    """
    logger.info("Iniciando proceso de entrenamiento del modelo")
    
    try:
        # Cargar datos
        logger.info("Cargando datos históricos")
        df = pd.read_csv('src/static/data/historical.csv')
        logger.info("Datos cargados correctamente")
        df = enriquecer_datos(df, logger)
        logger.info("Datos enriquecidos correctamente")
        realizar_eda(df, logger)
        logger.info("EDA realizado correctamente")

        features = obtener_features(logger)
        logger.info("Features obtenidos correctamente")
        
        X = df[features]
        y = df['target']
        
        split_idx = int(len(df) * 0.8)
        train_data = df.iloc[:split_idx]
        test_data = df.iloc[split_idx:]

        X_train = train_data[features]
        y_train = train_data['target']
        X_test = test_data[features]
        y_test = test_data['target']

        logger.info(f"Datos de entrenamiento: {len(X_train)} muestras")
        logger.info(f"Datos de prueba: {len(X_test)} muestras")
        logger.info(f"Fecha límite entrenamiento: {train_data['date'].max()}")
        logger.info(f"Fecha inicio prueba: {test_data['date'].min()}\n")

        # 1. ANÁLISIS DE LA DISTRIBUCIÓN DE CLASES
        logger.info("1. Análisis de distribución de clases:")
        logger.info(f"Train - Distribución: {y_train.value_counts().to_dict()}")
        logger.info(f"Test - Distribución: {y_test.value_counts().to_dict()}")
        logger.info(f"Train - Proporción clase 1: {y_train.mean():.3f}")
        logger.info(f"Test - Proporción clase 1: {y_test.mean():.3f}\n")

       
       
       # 2. ANÁLISIS DE CORRELACIONES ENTRE FEATURES
        logger.info("2. Análisis de correlaciones:")
        correlation_matrix = X_train.corr()
        logger.info("Correlaciones altas (>0.7) entre features:")
        high_corr = np.where(np.abs(correlation_matrix) > 0.7)
        for i, j in zip(high_corr[0], high_corr[1]):
            if i < j:  # Evitar duplicados
                logger.info(f"{correlation_matrix.index[i]} vs {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i,j]:.3f}")


        # Remover NaN
        df_clean = df.dropna()

        # Nueva división temporal
        split_idx = int(len(df_clean) * 0.8)
        train_enhanced = df_clean.iloc[:split_idx]
        test_enhanced = df_clean.iloc[split_idx:]

        X_train_enh = train_enhanced[features]
        y_train_enh = train_enhanced['target']
        X_test_enh = test_enhanced[features]
        y_test_enh = test_enhanced['target']

        logger.info(f"Datos después de feature engineering: {len(df_clean)} muestras")
        logger.info(f"Features adicionales creadas: {len(features)}\n")
     
        # 3. ENTRENAR MODELO CON FEATURES ADICIONALES (Se utilizan multiples modelos para comparar) y se selecciona el mejor

        logger.info("4. Comparando múltiples algoritmos: RandomForest, XGBoost y LogisticRegression")

        models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=15, 
        random_state=42),
            'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
        random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}

        for name, model in models.items():
            logger.info(f"\n--- {name} ---")
    
            # Entrenar modelo
            model.fit(X_train_enh, y_train_enh)

            # Predicciones
            y_pred = model.predict(X_test_enh)
            y_pred_proba = model.predict_proba(X_test_enh)[:, 1]

            # Métricas
            accuracy = accuracy_score(y_test_enh, y_pred)
            auc = roc_auc_score(y_test_enh, y_pred_proba)

            results[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'model': model
            }

            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"AUC: {auc:.4f}")

        # 5. MEJOR MODELO
        logger.info("\n5. Resumen de resultados:")
        best_model_name = max(results, key=lambda x: results[x]['auc'])
        best_model = results[best_model_name]['model']

        # 6. ANÁLISIS DE THRESHOLD ÓPTIMO
        logger.info(f"\n6. Optimización de threshold para {best_model_name}:")

        y_pred_proba_best = best_model.predict_proba(X_test_enh)[:, 1]

        # Calcular métricas de error para las probabilidades
        rmse = np.sqrt(mean_squared_error(y_test_enh, y_pred_proba_best))
        mae = mean_absolute_error(y_test_enh, y_pred_proba_best)
        
        logger.info("\nMétricas de error para probabilidades:")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"Interpretación:")
        logger.info(f"- RMSE: Error cuadrático medio de {rmse:.4f} (menor es mejor, rango 0-1)")
        logger.info(f"- MAE: Error absoluto medio de {mae:.4f} (menor es mejor, rango 0-1)")

        # Probar diferentes thresholds
        thresholds = np.arange(0.3, 0.8, 0.05)
        best_threshold = 0.5
        best_f1 = 0

        for threshold in thresholds:
           y_pred_thresh = (y_pred_proba_best >= threshold).astype(int)

           # Calcular F1-score
           f1 = f1_score(y_test_enh, y_pred_thresh)

           if f1 > best_f1:
               best_f1 = f1
               best_threshold = threshold

           logger.info(f"Threshold {threshold:.2f}: F1={f1:.3f}, Accuracy={accuracy_score(y_test_enh, y_pred_thresh):.3f}")
        
        logger.info(f"\nMejor threshold: {best_threshold:.2f} (F1={best_f1:.3f})")

        # Guardar el mejor modelo y el threshold óptimo
        model_data = {
            'model': best_model,
            'threshold': best_threshold,
            'model_name': best_model_name,
            'metrics': {
                'rmse': float(rmse),
                'mae': float(mae),
                'auc': float(results[best_model_name]['auc']),
                'accuracy': float(results[best_model_name]['accuracy']),
                'f1': float(best_f1)
            }
        }
        joblib.dump(model_data, 'src/static/models/model.pkl')
        
        logger.info(f"Modelo y threshold óptimo guardados exitosamente")
        logger.info(f"Mejor modelo: {best_model_name}")
        logger.info(f"Mejor AUC: {results[best_model_name]['auc']:.4f}")
        logger.info(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
        logger.info(f"Threshold óptimo: {best_threshold:.2f}")

        return df
    except Exception as e:
        logger.error(f"Error durante el entrenamiento del modelo: {str(e)}")
        raise

def predecir(logger: LoggerService):
    """
    Carga el modelo y realiza predicciones utilizando el threshold óptimo
    """
    logger.info("Iniciando proceso de predicción")
    
    try:
        # Cargar modelo y threshold óptimo
        logger.info("Cargando modelo y threshold óptimo")
        model_data = joblib.load('src/static/models/model.pkl')
        model = model_data['model']
        best_threshold = model_data['threshold']
        metrics = model_data.get('metrics', {})
        
        logger.info(f"Modelo cargado: {model_data['model_name']}")
        logger.info(f"Threshold óptimo: {best_threshold:.2f}")
        if metrics:
            logger.info("\nMétricas del modelo:")
            logger.info(f"RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            logger.info(f"MAE: {metrics.get('mae', 'N/A'):.4f}")
            logger.info(f"AUC: {metrics.get('auc', 'N/A'):.4f}")
            logger.info(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            logger.info(f"F1-Score: {metrics.get('f1', 'N/A'):.4f}")
        
        # Cargar datos más recientes
        logger.info("Cargando datos más recientes")
        df = pd.read_csv('src/static/data/historical.csv')
        df = enriquecer_datos(df, logger)
        
        # Preparar features para predicción
        logger.info("Preparando features para predicción")
        features = obtener_features(logger)
        X = df[features].iloc[-1:].copy()  # Última fila
        
        # Verificar si hay valores NaN
        if X.isna().any().any():
            logger.warning("Se detectaron valores NaN en los features. Se eliminarán las filas con NaN.")
            X = X.dropna()
            if len(X) == 0:
                raise ValueError("No hay datos válidos para realizar la predicción después de eliminar NaN")
        
        # Realizar predicción
        logger.info("Realizando predicción")
        probabilidad = model.predict_proba(X)[:, 1]
        prediccion = (probabilidad >= best_threshold).astype(int)
        
        resultado = {
            'prediccion': int(prediccion[0]),
            'probabilidad': float(probabilidad[0]),
            'threshold': float(best_threshold),
            'fecha': df['date'].iloc[-1],
            'precio_actual': float(df['close'].iloc[-1]),
            'metricas_modelo': metrics
        }

        
        logger.info(f"Predicción completada: {resultado}")
        return resultado
        
    except Exception as e:
        error_msg = f"Error durante la predicción: {str(e)}"
        logger.error(error_msg)
        return {'error': str(e)}

def entrenar_arima(logger: LoggerService):
    """
    Entrena el modelo ARIMA y lo guarda como modelo_arima.pkl
    """
    logger.info("Iniciando proceso de entrenamiento del modelo ARIMA")
    
    try:
        # Cargar datos
        logger.info("Cargando datos")
        data = pd.read_csv('src/static/data/historical.csv', index_col="date")
        data = data.sort_values(by='date')

        logger.info("Dividiendo datos en entrenamiento y prueba")
        train_size = int(len(data) * 0.8)
        train, test = data.iloc[:train_size], data.iloc[train_size:]
        train_original = train.copy()

        logger.info("Verificando estacionariedad de la serie")
        result = adfuller(train["close"])

        d = 0
        while result[1] > 0.05:
            train["close"] = train["close"].diff()
            d += 1
            result = adfuller(train["close"].dropna())

        print(f"Serie de entrenamiento es estacionaria después de {d} diferenciaciones.")

        p = 1
        q = 1
        model = ARIMA(train["close"].dropna(), order=(p, d, q))
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=len(test))

        last_value = train_original["close"].iloc[-1]  # asegúrate de tener una copia de los datos originales

        # Reconstruir los valores a partir del pronóstico de diferencias
        forecast_values = forecast.cumsum() + last_value

        logger.info("Calculando métricas")
        # RMSE
        rmse = np.sqrt(mean_squared_error(test["close"], forecast_values))

        # MAE
        mae = mean_absolute_error(test["close"], forecast_values)

        logger.info(f"RMSE: {rmse:.2f}")
        logger.info(f"MAE: {mae:.2f}")

        
        # Guardar modelo ARIMA, si existe se sobreescribe
        logger.info("Guardando modelo ARIMA")
        with open("src/static/models/modelo_arima.pkl", "wb") as f:
            pickle.dump(fitted_model, f)

        # Guardar métricas
        logger.info("Guardando métricas")
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "p": p,
            "d": d,
            "q": q
        }
        with open("src/static/models/metrics_arima.pkl", "wb") as f:
            pickle.dump(metrics, f)
        logger.info("Modelo ARIMA entrenado y guardado exitosamente")
    except Exception as e:
        logger.error(f"Error durante el entrenamiento del modelo ARIMA: {str(e)}")
        raise

def main():
    logger = LoggerService()
    entrenar(logger)
    predecir(logger)
    entrenar_arima(logger)

if __name__ == '__main__':
    main()