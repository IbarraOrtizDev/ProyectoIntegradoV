<div style="text-align:center">
<img style="width:300px" src="https://www.iudigital.edu.co/images/11.-IU-DIGITAL.png"/>
</div>

<br/>

<div style="text-align:center">
 <h2>Proyecto Integrado V - Linea de Énfasis (Entrega 1)</h2>
</div>

## Integrantes
<br/>
<div style="text-align:center">
 <h3>Edwin Alexander Ibarra Ortiz</h3>
 <h3>PREICA2501B020128</h3>
</div>
<br/>
<div style="text-align:center">
 <h3>Sergio Andres Rios Gomez</h3>
 <h3>PREICA2501B020128</h3>
</div>
<br/>
<br/>

<div style="text-align:center">
 <h3>IU Digital de Antioquia</h3>
 <h3>Ingeniería de Software y Datos</h3>
 <h3>2025</h3>
</div>


# Recolección Automatizada de Datos Históricos: Taiwan Semiconductor Manufacturing Company (TSM)

## Introducción

El presente proyecto tiene como objetivo principal demostrar la capacidad de automatizar la recolección continua de datos históricos asociados a un indicador económico distinto de Yahoo Finanzas. En esta entrega se ha elegido como indicador el comportamiento bursátil de Taiwan Semiconductor Manufacturing Company (TSMC), una de las empresas más relevantes a nivel mundial en la industria de semiconductores. La información recolectada proporciona una base sólida para análisis financieros, estudios de mercado y predicciones de comportamiento económico.

TSMC desempeña un papel estratégico en la economía global, siendo un referente para entender la evolución tecnológica y el impacto de la cadena de suministro en sectores como la electrónica de consumo, la automoción y la inteligencia artificial. Por ello, mantener un historial actualizado de su desempeño financiero resulta relevante tanto para el análisis académico como para la toma de decisiones informadas en entornos profesionales.

Este proyecto ha sido desarrollado bajo un enfoque de programación orientada a objetos (OOP), asegurando modularidad, mantenibilidad y trazabilidad. Además, la automatización del proceso de recolección de datos se implementa utilizando GitHub Actions, configurado para ejecutarse diariamente a las 00:00 horas, garantizando así la persistencia del histórico sin sobrescribir registros anteriores.



## Características

- Recolección automática de datos de Yahoo Finance
- Almacenamiento en base de datos SQLite
- Exportación a archivos CSV
- Sistema de logs diarios
- Ejecución programada mediante GitHub Actions
- Manejo de errores y excepciones
- Validación de datos duplicados

## Requisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Conexión a Internet

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/IbarraOrtizDev/IbarraOrtiz_EdwinAlexander_infraestructura-arquitectura-big-data.git
cd ProyectoIntegradoV
```

2. Crear y activar entorno virtual:
```bash
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/Mac:
source .venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -e .
```

## Configuración

1. Crear archivo `.env` en la raíz del proyecto con las siguientes variables:
```env
URL=https://query1.finance.yahoo.com/v8/finance/chart/TSM?events=capitalGain%7Cdiv%7Csplit&formatted=true&includeAdjustedClose=true&interval=1d&period1=[0]&period2=[1]&symbol=TSM&userYfid=true&lang=en-US&region=US
DB_PATH=src/static/historical.db
SAMPLE_FILE_PATH=src/static/historica.csv
LOGS_DIR_PATH=src/logs
NAME_TABLE=stock_data
```

## Uso

1. Ejecutar el script principal:
```bash
python src/collector.py
```

2. Los datos se almacenarán en:
   - Base de datos: `src/static/historical.db`
   - Archivo CSV: `src/static/historica.csv`
   - Logs: `src/logs/log_YYYY-MM-DD.txt`

## Estructura del Proyecto

```
ProyectoIntegradoV/
├── src/
│   ├── collector.py          # Script principal
│   ├── logger.py            # Servicio de logs
│   ├── infraestructure/
│   │   └── manageDataBase.py # Gestión de base de datos
│   ├── service/
│   │   ├── queryYahooAdapter.py    # Adaptador para Yahoo Finance
│   │   └── parseDataYahooQuery.py  # Parser de datos
│   ├── static/
│   ├   |── data/
|   │   │   ├── historical.db    # Base de datos SQLite
|   │   │   └── historica.csv    # Archivo CSV
│   └── logs/                # Directorio de logs
├── .env                     # Variables de entorno
├── setup.py                 # Configuración del paquete
└── README.md               # Este archivo
```

## Ejecución Automática

El proyecto está configurado para ejecutarse automáticamente:
- Diariamente a medianoche mediante GitHub Actions
- En cada push a la rama main
- En cada pull request a la rama main

## Logs

Los logs se generan diariamente en el directorio `src/logs` con el formato:
```
YYYY-MM-DD HH:MM:SS - LEVEL - Mensaje
```

Niveles de log disponibles:
- INFO: Información general
- ERROR: Errores y excepciones
- WARNING: Advertencias
- DEBUG: Información de depuración

## Contribución

1. Fork el repositorio
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## Autores

- Edwin Alexander Ibarra - [edwin.ibarra@est.iudigital.edu.co](mailto:edwin.ibarra@est.iudigital.edu.co)
- Sergio Rios - [sergio.rios@est.iudigital.edu.co](mailto:sergio.rios@est.iudigital.edu.co)

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.
