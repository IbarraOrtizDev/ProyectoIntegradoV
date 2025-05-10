#!/bin/bash

# Salir si ocurre un error
set -e

# Crear entorno virtual
python3 -m venv .venv

# Activar entorno virtual
source .venv\\Scripts\\activate

# Actualizar pip e instalar el paquete local
python -m pip install --upgrade pip
pip install .

# Crear archivo .env con las variables necesarias
cat <<EOL > .env
URL=https://query1.finance.yahoo.com/v8/finance/chart/TSM?events=capitalGain%7Cdiv%7Csplit&formatted=true&includeAdjustedClose=true&interval=1d&period1=[0]&period2=[1]&symbol=TSM&userYfid=true&lang=en-US&region=US
DB_PATH=src/static/historical.db
SAMPLE_FILE_PATH=src/static/historica.csv
LOGS_DIR_PATH=src/logs
EOL

# Ejecutar el script Python
python src/collector.py
