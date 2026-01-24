Sistema de Análisis y Predicción de Tendencias Científicas (PLN + ML)

Este proyecto implementa un sistema completo de análisis y monitoreo de tendencias en investigación científica, combinando Procesamiento de Lenguaje Natural (PLN) y Machine Learning (ML) sobre fuentes académicas abiertas.

El sistema opera en dos modos:
Modo Histórico: análisis longitudinal basado en un dataset consolidado (MIT).
Modo Live (Tiempo casi real): monitoreo dinámico usando datos recientes de arXiv (y RSS cuando está disponible), sin costo y sin depender de APIs privadas.


Objetivo del proyecto
Detectar, analizar y visualizar tendencias emergentes, consolidadas y en declive dentro de la investigación científica, con énfasis en áreas de Ciencias de la Computación e Inteligencia Artificial, permitiendo:
Exploración histórica de tendencias.
Observación de actividad científica reciente.
Interpretación visual clara mediante dashboards interactivos.

Requisitos del sistema
Software: Python 3.10+
Sistema operativo: Windows / Linux / macOS
Navegador web moderno

Librerías principales
streamlit
pandas
scikit-learn
matplotlib
wordcloud
feedparser
requests
pyarrow
certifi


Instalación de dependencias:pip install -r requirements.txt

Cómo ejecutar el sistema
Opción 1: Ejecución directa (recomendada para evaluación)
Desde la raíz del proyecto: streamlit run app/streamlit_app.py

Opción 2: Ejecución con script (Windows)
En la carpeta scripts/: run_dashboard.bat
Esta opción evita el uso manual de comandos y está pensada para demostraciones.


Funcionamiento del pipeline Live
El modo Live NO requiere que el sistema esté encendido todo el tiempo.

Cada vez que se presiona “Actualizar ahora”:
Se consultan nuevas publicaciones desde arXiv.
Los datos se normalizan y limpian.
Se agregan al dataset Live existente.
El dashboard se recalcula automáticamente.

Los periodos temporales aumentan únicamente cuando existen datos de días distintos, no por tiempo de ejecución continuo.

Qué esperar del dashboard

1.- Indicadores principales (KPIs)
Total de documentos analizados.
Cobertura temporal del dataset.
Número de periodos disponibles (día, semana o mes).

2.- Nube de conceptos
Visualización de términos y frases relevantes.
Basada en TF-IDF + N-grams.
Ajustable en complejidad desde la interfaz.

3️.- Modo Actividad (cuando la historia es corta)
Ranking de términos más activos.
Actividad diaria de publicaciones.
Tabla de artículos recientes.

4️.- Modo Tendencias (cuando hay suficiente historia)
Ranking de tendencias detectadas.
Visualización temporal del término dominante.
Interpretación directa del comportamiento científico reciente.
El sistema nunca queda vacío: siempre muestra información relevante según el contexto temporal disponible.

Consideraciones técnicas importantes
Los archivos .parquet no se versionan (por diseño).
El sistema es reproducible: los datos se regeneran localmente.
La lógica Live usa ventanas temporales dinámicas (no simulación).
El dashboard incluye manejo de errores y mensajes interpretables.
