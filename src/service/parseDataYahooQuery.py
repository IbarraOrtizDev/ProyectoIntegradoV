from logger import LoggerService
import pandas as pd
from datetime import datetime, timezone

class ParseDataYahooQuery:
    def __init__(self, data: dict):
        self.data = data
        self.logger = LoggerService()
        self.gmtoffset = data["chart"]["result"][0]["meta"]["gmtoffset"]
        self.symbol = data["chart"]["result"][0]["meta"]["symbol"]
        self.currency = data["chart"]["result"][0]["meta"]["currency"]

    def parse(self):
        try:
            self.logger.info("Iniciando el parseo de los datos")
            data = self.data["chart"]["result"][0]
            timestamp = data["timestamp"]
            open = data["indicators"]["quote"][0]["open"]
            close = data["indicators"]["quote"][0]["close"]
            low = data["indicators"]["quote"][0]["low"]
            high = data["indicators"]["quote"][0]["high"]
            volume = data["indicators"]["quote"][0]["volume"]
            adjclose = data["indicators"]["adjclose"][0]["adjclose"]
            df = self.parse_data(timestamp, open, close, low, high, volume, adjclose)
            return df
        except Exception as e:
            self.logger.error(f"Error al parsear los datos: {e}")
            raise

    def parse_data(self, timestamp: list, open: list, close: list, low: list, high: list, volume: list, adjclose: list):
        try:
            # Convertir los datos a un dataframe
            df = pd.DataFrame({
                "timestamp": timestamp,
                "open": open,
                "close": close,
                "low": low,
                "high": high,
                "volume": volume,
                "adjclose": adjclose
            })
            # Convertir el timestamp a un datetime
            df["date"] = df["timestamp"].apply(self.parseDataToJson)
            df["symbol"] = self.symbol
            df["currency"] = self.currency
            return df
        except Exception as e:
            self.logger.error(f"Error al parsear los datos: {e}")
            raise
    
    def parseDataToJson(self, timestamp: int):
        try:
            dt = datetime.fromtimestamp(timestamp + self.gmtoffset, tz=timezone.utc)
            return dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        except Exception as e:
            self.logger.error(f"Error al parsear los datos: {e}")
            raise
    
    def get_data(self):
        try:
            df = self.parse_data(self.data)
            pass
        except Exception as e:
            self.logger.error(f"Error al obtener los datos: {e}")
            raise

