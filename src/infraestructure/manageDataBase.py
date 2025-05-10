from logger import LoggerService
import os
import sqlite3
import pandas as pd

class ManageDataBase:
    def __init__(self):
        self.logger = LoggerService()
        self.db_path = os.getenv("DB_PATH")
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.table_name = os.getenv("NAME_TABLE")
        self.create_table()

    def create_table(self):
        try:
            self.logger.info("Creando la tabla en la base de datos")
            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adjclose REAL,
                    date TEXT,
                    symbol TEXT,
                    currency TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            self.cursor.execute(create_table_query)
            self.conn.commit()
            self.logger.info("Tabla creada exitosamente")
        except Exception as e:
            self.logger.error(f"Error al crear la tabla en la base de datos: {e}")
            raise

    def insert_data(self, data: pd.DataFrame):
        self.logger.info("Insertando los datos en la base de datos")
        try:
            print(data.head())
            print(data.columns)
            
            # Obtener el último timestamp de la base de datos
            last_timestamp = self.get_data(f"SELECT MAX(timestamp) FROM {self.table_name}")
            
            # Si hay datos en la base de datos, filtrar los nuevos
            if last_timestamp and last_timestamp[0][0] is not None:
                self.logger.info(f"Filtrando datos posteriores a {last_timestamp[0][0]}")
                data = data[data['timestamp'] > last_timestamp[0][0]]
            else:
                self.logger.info("No hay datos previos en la base de datos, insertando todos los registros")
            
            # Si no hay datos nuevos para insertar, terminar
            if data.empty:
                self.logger.info("No hay nuevos datos para insertar")
                return
            
            # Convertir el DataFrame a una lista de tuplas
            columns = data.columns.tolist()
            values = data[columns].values.tolist()
            
            # Preparar la consulta SQL
            placeholders = ','.join(['?' for _ in columns])
            query = f"INSERT INTO {self.table_name} ({','.join(columns)}) VALUES ({placeholders})"
            
            # Ejecutar la inserción
            self.cursor.executemany(query, values)
            self.conn.commit()
            self.logger.info("Datos insertados exitosamente")
        except Exception as e:
            self.logger.error(f"Error al insertar los datos en la base de datos: {e}")
            raise

    def get_data(self, query: str):
        self.logger.info("Obteniendo los datos de la base de datos")
        try:
            self.cursor.execute(query)
            return self.cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Error al obtener los datos de la base de datos: {e}")
            raise

    def get_data_df(self, query: str):
        self.logger.info("Obteniendo los datos de la base de datos")
        try:
            return pd.read_sql_query(query, self.conn)
        except Exception as e:
            self.logger.error(f"Error al obtener los datos de la base de datos: {e}")
            raise


