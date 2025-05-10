@echo off
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install .
echo [DEFAULT] > .env
echo URL=https://query1.finance.yahoo.com/v8/finance/chart/TSM?events=capitalGain%7Cdiv%7Csplit&formatted=true&includeAdjustedClose=true&interval=1d&period1=[0]&period2=[1]&symbol=TSM&userYfid=true&lang=en-US&region=US >> .env
echo DB_PATH=src/static/historical.db >> .env
echo SAMPLE_FILE_PATH=src/static/historica.csv >> .env
echo LOGS_DIR_PATH=src/logs >> .env

python src/collector.py