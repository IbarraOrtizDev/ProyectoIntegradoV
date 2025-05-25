from src.logger import LoggerService
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def realizar_eda(df, logger: LoggerService, save_path='src/static/eda/'):
    """
    Realiza análisis exploratorio de datos y guarda visualizaciones
    """
    logger.info("Iniciando análisis exploratorio de datos (EDA)")
    
    try:
        # Crear directorio si no existe
        logger.info(f"Creando directorio para visualizaciones: {save_path}")
        os.makedirs(save_path, exist_ok=True)
        
        # Distribución de la variable objetivo
        logger.info("Generando gráfico de distribución de la variable objetivo")
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='target')
        plt.title('Distribución de Movimientos de Precio')
        plt.savefig(f'{save_path}distribucion_target.png')
        plt.close()
        
        # Correlación entre variables
        logger.info("Generando matriz de correlación")
        plt.figure(figsize=(12, 8))
        correlation_matrix = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        plt.savefig(f'{save_path}correlacion.png')
        plt.close()
        
        # Series temporales
        logger.info("Generando gráfico de serie temporal")
        plt.figure(figsize=(15, 6))
        plt.plot(df['date'], df['close'])
        plt.title('Evolución del Precio de Cierre')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}serie_temporal.png')
        plt.close()
        
        # Guardar estadísticas descriptivas
        logger.info("Guardando estadísticas descriptivas")
        stats = df.describe()
        stats.to_csv(f'{save_path}estadisticas_descriptivas.csv')
        
        logger.info("EDA completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error durante el EDA: {str(e)}")
        raise