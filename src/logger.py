import logging
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

class LoggerService:
    def __init__(self):
        """
        Inicializa el servicio de logs.
        La ruta del directorio de logs se toma de la variable de entorno LOGS_DIR_PATH.
        Si no está definida, se usa 'src/logs' como valor por defecto.
        """
        # Cargar variables de entorno
        load_dotenv()
        
        # Obtener la ruta de logs desde la variable de entorno
        self.logs_dir_path = os.getenv('LOGS_DIR_PATH', 'src/logs')
        print(f"Ruta de logs configurada: {self.logs_dir_path}")
        
        # Asegurarse de que el directorio de logs existe
        self._ensure_logs_directory()
        
        # Configurar el logger
        self._setup_logger()

    def _ensure_logs_directory(self):
        """Asegura que el directorio de logs existe."""
        try:
            # Crear el directorio de logs si no existe
            Path(self.logs_dir_path).mkdir(parents=True, exist_ok=True)
            print(f"Directorio de logs creado/verificado en: {self.logs_dir_path}")
        except Exception as e:
            print(f"Error al crear el directorio de logs: {e}")
            raise

    def _setup_logger(self):
        """Configura el logger con el formato y archivo de salida adecuados."""
        try:
            # Obtener la fecha actual para el nombre del archivo
            current_date = datetime.now().strftime("%Y-%m-%d")
            log_file = os.path.join(self.logs_dir_path, f"log_{current_date}.txt")
            print(f"Archivo de log configurado: {log_file}")
            
            # Configurar el logger
            self.logger = logging.getLogger("ProyectoIntegradoV")
            self.logger.setLevel(logging.INFO)
            
            # Crear el manejador de archivo
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # Definir el formato del log
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            # Agregar el manejador al logger
            self.logger.addHandler(file_handler)
            print("Logger configurado exitosamente")
            
        except Exception as e:
            print(f"Error al configurar el logger: {e}")
            raise

    def info(self, message: str):
        """Registra un mensaje de nivel INFO."""
        self.logger.info(message)

    def error(self, message: str):
        """Registra un mensaje de nivel ERROR."""
        self.logger.error(message)

    def warning(self, message: str):
        """Registra un mensaje de nivel WARNING."""
        self.logger.warning(message)

    def debug(self, message: str):
        """Registra un mensaje de nivel DEBUG."""
        self.logger.debug(message)


if __name__ == "__main__":
    logger = LoggerService()
    logger.info("Iniciando el servicio de logs")
    logger.error("Este es un mensaje de error de prueba")
    logger.warning("Este es un mensaje de advertencia de prueba")
    logger.debug("Este es un mensaje de debug de prueba")
