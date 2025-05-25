import pandas as pd
import numpy as np
from logger import LoggerService

def enriquecer_datos(df, logger: LoggerService):
    """
    Enrich the data with additional variables and transformations.
    Args:
        df (pd.DataFrame): DataFrame with historical data
    Returns:
        pd.DataFrame: DataFrame enriched with new features
    """
    logger.info("Starting data enrichment process")
    
    try:
        # Convertir fecha a datetime si no lo es (solo fecha, no hora)
        logger.info("Convirtiendo columna date a datetime")
        df['date'] =  pd.to_datetime(df['date'])
        
        # Variables temporales
        logger.info("Generando variables temporales")
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        
        # Retornos
        logger.info("Calculando retornos")
        df['returns'] = df['close'].pct_change()
        df['returns_prev_day'] = df['returns'].shift(1)
        
        # Volatilidad móvil (20 días)
        logger.info("Calculando volatilidad móvil de 20 días")
        df['volatility_20d'] = df['returns'].rolling(window=20).std()
        
        # Medias móviles
        logger.info("Calculando medias móviles")
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI (Relative Strength Index)
        logger.info("Calculando RSI")
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Variable objetivo
        logger.info("Generando variable objetivo")
        df['target'] = (df['close'] > df['close'].shift(1)).astype(int)
        
        # Eliminar filas con NaN
        logger.info("Eliminando filas con valores NaN")
        df_original_len = len(df)
        df = df.dropna()
        df_final_len = len(df)
        logger.info(f"Filas eliminadas: {df_original_len - df_final_len}")
        
        logger.info("Proceso de enriquecimiento completado exitosamente")
        df = df.sort_values('date').reset_index(drop=True)
        return create_enhanced_features(df)
        
    except Exception as e:
        logger.error(f"Error en el proceso de enriquecimiento: {str(e)}")
        raise



# 3. CREAR FEATURES ADICIONALES MÁS PREDICITIVAS
# Dado que el modelo inicial no es bueno, se crea un nuevo modelo con features adicionales
# Estas features son:
# - SMA ratio
# - SMA cross
# - Volatility ratio
# - Volume ratio
# - RSI oversold
# - RSI overbought
# - Returns 2d
# - Returns 5d
# Estas features se crean para mejorar la precisión del modelo
def create_enhanced_features(df):
    """Crear features adicionales más predictivas"""
    df_enhanced = df.copy()
    
    # Features de momentum
    df_enhanced['sma_ratio'] = (df['close'] - df['sma_20']) / df['sma_20']
    df_enhanced['sma_cross'] = np.where(df['sma_20'] > df['sma_50'], 1, 0)
    
    # Features de volatilidad
    df_enhanced['volatility_ratio'] = df['volatility_20d'] / df['volatility_20d'].rolling(60).mean()
    
    # Features de volumen (si disponible)
    if 'volume' in df.columns:
        df_enhanced['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Features de RSI
    df_enhanced['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)
    df_enhanced['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
    
    # Features de returns múltiples
    df_enhanced['returns_2d'] = df['close'].pct_change(2)
    df_enhanced['returns_5d'] = df['close'].pct_change(5)
    
    return df_enhanced


def obtener_features(logger: LoggerService):
    """
    Retorna la lista de features utilizadas para el modelo.
    
    Returns:
        list: Lista de nombres de features
    """
    logger.info("Obteniendo lista de features para el modelo")
    features = [
        'returns_prev_day',
        'volatility_20d',
        'sma_20',
        'sma_50',
        'rsi',
        'day_of_week',
        'month',
        'quarter',

        # Features adicionales creadas para mejorar el modelo
        'sma_ratio',
        'sma_cross',
        'volatility_ratio',
        'rsi_oversold',
        'rsi_overbought',
        'returns_2d',
        'returns_5d'
    ]
    logger.info(f"Features seleccionadas: {', '.join(features)}")
    return features