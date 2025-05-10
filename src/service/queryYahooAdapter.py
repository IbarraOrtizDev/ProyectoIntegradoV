from logger import LoggerService
import requests
from dotenv import load_dotenv
import os
from infraestructure.manageDataBase import ManageDataBase
from datetime import datetime
import time


class QueryYahooAdapter:
    def __init__(self):
        load_dotenv()   
        self.url = os.getenv("URL")
        self.logger = LoggerService()
        self.manageDataBase = ManageDataBase()
        self.table_name = os.getenv("NAME_TABLE")

    def get_data(self):
        try:
            self.logger.info("Iniciando la consulta de datos de Yahoo Finance")
            url = self.config_query()
            headers = {
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "accept-language": "es-ES,es;q=0.9",
                "priority": "u=0, i",
                "sec-ch-ua": "\"Chromium\";v=\"136\", \"Google Chrome\";v=\"136\", \"Not.A/Brand\";v=\"99\"",
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": "\"Windows\"",
                "sec-fetch-dest": "document",
                "sec-fetch-mode": "navigate",
                "sec-fetch-site": "none",
                "sec-fetch-user": "?1",
                "upgrade-insecure-requests": "1",
                "user-agent": "Mozilla/5.0",
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            self.logger.info("Datos obtenidos exitosamente")
            return response.json()
        except Exception as e:
            self.logger.error(f"Error al obtener datos de la API: {e}")
            raise

    def config_query(self):
        try:
            self.logger.info("Configurando la consulta")
            self.url = os.getenv("URL")
            response = self.manageDataBase.get_data(f"select timestamp from {self.table_name} where symbol = 'TSM' order by timestamp desc limit 1")
            if response and response[0][0] is not None:
                self.logger.info("Datos encontrados en la base de datos")
                self.replace_url(response[0][0], int(time.time()))
            else:
                self.logger.info("No se encontraron datos en la base de datos")
                # Datos desde 1997
                start_date = int(datetime(2024, 1, 1).timestamp())
                #end_date = int(time.time())
                end_date = int(datetime(2025, 1, 1).timestamp())
                self.logger.info(f"Datos desde {start_date} hasta {end_date}")
                self.replace_url(start_date, end_date)

            return self.url
        except Exception as e:
            self.logger.error(f"Error al configurar la consulta: {e}")
            raise

    def replace_url(self, start_date: int, end_date: int):
        try:
            self.url = self.url.replace("[0]", str(start_date))
            self.url = self.url.replace("[1]", str(end_date))
        except Exception as e:
            self.logger.error(f"Error al reemplazar la URL: {e}")
            raise
