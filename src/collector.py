from dotenv import load_dotenv
from logger import LoggerService
from infraestructure.manageDataBase import ManageDataBase
from service.parseDataYahooQuery import ParseDataYahooQuery
from service.queryYahooAdapter import QueryYahooAdapter
import os
from datetime import datetime

def main():
    # Cargar variables de entorno
    load_dotenv()
    
    # Inicializar el logger
    logger = LoggerService()
    
    try:
        manageDataBase = ManageDataBase()
        queryYahooAdapter = QueryYahooAdapter()
        data = queryYahooAdapter.get_data()
        parseDataYahooQuery = ParseDataYahooQuery(data)
        data = parseDataYahooQuery.parse()
        manageDataBase.insert_data(data)
        fecha_actual = datetime.now().strftime("%Y-%m-%d")
        data.to_csv(os.getenv("SAMPLE_FILE_PATH") + f"_{fecha_actual}.csv", index=False)

        # All data
        data_all = manageDataBase.get_data_df(f"SELECT * FROM {os.getenv('NAME_TABLE')}")
        # Remove old file
        if os.path.exists(os.getenv("SAMPLE_FILE_PATH") + ".csv"):
            os.remove(os.getenv("SAMPLE_FILE_PATH") + ".csv")
        data_all.to_csv(os.getenv("SAMPLE_FILE_PATH") + ".csv", index=False)
        
    except Exception as e:
        logger.error(f"Error durante la recolección de datos: {str(e)}")
        raise

if __name__ == '__main__':
    main()