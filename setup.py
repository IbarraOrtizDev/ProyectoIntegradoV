from setuptools import setup, find_packages # type: ignore

setup(
    name='proyecto-integrado-V',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'requests',
        'pandas',
        'openpyxl',
        'python-dotenv'
    ],
    entry_points={
        'console_scripts': [
            'collector=collector:main',
        ],
    },
    include_package_data=True,
    description='Proyecto Integrado V -> Consumir un servicio de Yahoo Finance y almacenar la información en un sqlite3 y en un .csv',
    long_description=open('README.md').read(),
    author='Edwin Alexander Ibarra - Sergio Rios',
    author_email='edwin.ibarra@est.iudigital.edu.co - sergio.rios@est.iudigital.edu.co',
    url='https://github.com/IbarraOrtizDev/IbarraOrtiz_EdwinAlexander_infraestructura-arquitectura-big-data',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)