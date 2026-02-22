# Penguin Classifier — MLOps Taller 1

Proyecto de clasificación de especies de pingüinos usando modelos de Machine Learning, desplegado con Docker Compose. Incluye un entorno Jupyter para entrenamiento y una API FastAPI para inferencia en tiempo real.

## Arquitectura

```
┌─────────────────┐       volúmenes compartidos       ┌─────────────────┐
│    Jupyter Lab   │ ──── models / report / data ────▶ │   FastAPI (API)  │
│   (puerto 8888)  │                                   │   (puerto 8000)  │
└─────────────────┘                                    └─────────────────┘
```

Ambos servicios comparten volúmenes Docker para modelos, reportes, datos y resultados.

## Estructura del Proyecto

```
MLOps_Taller1/
├── API/
│   └── app.py                          # API FastAPI de inferencia
├── data/
│   └── penguins_v1.csv                 # Dataset de pingüinos
├── Docker/
│   ├── docker-compose.yml              # Orquestación de servicios
│   ├── Dockerfile.api                  # Imagen de la API
│   └── Dockerfile.jupyter              # Imagen de Jupyter
└── jupyter/
    └── notebooks/
        ├── train.ipynb                 # Notebook de entrenamiento
        └── utils/
            └── model_trainer.py        # Clase ModelTrainer
```

## Requisitos Previos

- [Docker](https://docs.docker.com/get-docker/) y [Docker Compose](https://docs.docker.com/compose/install/)

## Diseño del Docker Compose

El archivo `Docker/docker-compose.yml` orquesta dos servicios que se comunican a través de volúmenes compartidos. A continuación se explica cada decisión de diseño.

### Contexto de build

Ambos servicios usan `context: ..` (la raíz del proyecto) como contexto de build, porque los Dockerfiles necesitan acceder a carpetas hermanas (`API/`, `data/`, `jupyter/`):

```yaml
services:
  jupyter:
    build:
      context: ..                        # raíz del proyecto
      dockerfile: Docker/Dockerfile.jupyter
  api:
    build:
      context: ..
      dockerfile: Docker/Dockerfile.api
```

Esto permite que cada Dockerfile copie archivos desde cualquier carpeta del proyecto sin necesidad de mover archivos.

### Servicio Jupyter

```yaml
jupyter:
  ports:
    - "8888:8888"
  volumes:
    - shared_models:/app/models
    - shared_report:/app/report
    - shared_data:/app/data
    - shared_results:/app/results
  environment:
    - JUPYTER_TOKEN=mlops2024
```

- Expone el puerto `8888` para acceder a Jupyter Lab desde el navegador.
- Monta 4 volúmenes para que los artefactos generados (modelos `.pkl`, métricas, logs) persistan y sean accesibles por la API.
- El token de acceso se configura vía variable de entorno.

### Servicio API

```yaml
api:
  ports:
    - "8000:8000"
  volumes:
    - shared_models:/app/modelos
    - shared_report:/app/report
    - shared_results:/app/results
  depends_on:
    - jupyter
```

- Expone el puerto `8000` para recibir peticiones HTTP.
- Monta los mismos volúmenes de modelos y reportes (nota: el punto de montaje es `/app/modelos` porque así lo espera `app.py`).
- `depends_on: jupyter` asegura que el contenedor de Jupyter se inicie primero, aunque no garantiza que los modelos estén entrenados — eso requiere ejecución manual del notebook.

### Volúmenes nombrados

```yaml
volumes:
  shared_models:    # Pipelines .pkl entrenados
  shared_report:    # model_metrics.pkl con métricas
  shared_data:      # Dataset penguins_v1.csv
  shared_results:   # predictions.log
```

Se usan volúmenes nombrados para:
- Persistir datos entre reinicios de contenedores (`docker-compose down` conserva los volúmenes).
- Compartir artefactos entre servicios sin exponer rutas del host.
- Limpiar todo con `docker-compose down -v` cuando se quiera empezar desde cero.

### Dockerfile.jupyter

```dockerfile
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
WORKDIR /app
RUN uv init --no-readme && \
    uv add jupyterlab pandas numpy scikit-learn matplotlib seaborn joblib
COPY data/penguins_v1.csv /app/data/penguins_v1.csv
COPY jupyter/notebooks/ /app/notebooks/
RUN mkdir -p /app/models
EXPOSE 8888
CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", "--NotebookApp.token=mlops2024"]
```

- Usa `uv` como gestor de paquetes (más rápido que pip).
- Copia el dataset y los notebooks dentro de la imagen para que estén disponibles sin volúmenes adicionales.
- El dataset también se monta como volumen, pero la copia en la imagen sirve como fallback.

### Dockerfile.api

```dockerfile
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
WORKDIR /app
RUN uv init --no-readme && \
    uv add fastapi uvicorn scikit-learn joblib numpy pydantic pandas
COPY API/app.py /app/app.py
RUN mkdir -p /app/modelos /app/report /app/results
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", \
     "--port", "8000", "--reload"]
```

- Instala solo las dependencias necesarias para servir predicciones (sin matplotlib, seaborn ni jupyterlab).
- Crea los directorios que se montarán como volúmenes para evitar errores si los volúmenes están vacíos.
- `--reload` permite que uvicorn detecte cambios en `app.py` durante desarrollo.

### Flujo de comunicación entre servicios

```
1. docker-compose up --build
   ├── Construye imagen jupyter (python + jupyterlab + sklearn)
   └── Construye imagen api (python + fastapi + sklearn)

2. Jupyter arranca primero (api depends_on jupyter)

3. Usuario abre http://localhost:8888 y ejecuta train.ipynb
   └── ModelTrainer guarda pipelines en /app/models/ (volumen shared_models)
   └── ModelTrainer guarda métricas en /app/report/ (volumen shared_report)

4. API en http://localhost:8000 lee los mismos volúmenes
   └── GET /models → descubre *_model.pkl en /app/modelos/
   └── POST /classify/{model} → carga pipeline, predice, loguea en /app/results/
```

## Construcción y Despliegue

### 1. Construir y levantar los servicios

```bash
cd MLOps_Taller1/Docker
docker-compose up --build
```

Esto construye ambas imágenes y levanta:
- Jupyter Lab en `http://localhost:8888` (token: `mlops2024`)
- API en `http://localhost:8000`

### 2. Reconstruir solo un servicio

```bash
docker-compose up --build api       # solo reconstruye la API
docker-compose up --build jupyter   # solo reconstruye Jupyter
```

### 3. Detener los servicios

```bash
docker-compose down
```

Para eliminar también los volúmenes (modelos, reportes, etc.) y empezar desde cero:

```bash
docker-compose down -v
```

## Entrenamiento de Modelos (Creación desde Cero)

Cuando los volúmenes están vacíos (primera ejecución o después de `docker-compose down -v`), es necesario entrenar los modelos antes de usar la API.

### Paso a paso

1. Acceder a Jupyter Lab en `http://localhost:8888` con el token `mlops2024`.

2. Abrir el notebook `notebooks/train.ipynb`.

3. Ejecutar todas las celdas en orden. El notebook realiza:

   - **Carga de datos**: Lee `penguins_v1.csv` con 7 features originales (island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, year).
   - **Limpieza**: Verifica nulos y duplicados, elimina la columna `id`.
   - **Feature engineering**: Crea `bill_ratio` (bill_length / bill_depth) y `body_mass_kg` (body_mass_g / 1000), resultando en 9 features totales.
   - **Split**: 80% train / 20% test con estratificación (`random_state=42`).
   - **Entrenamiento**: Tres modelos como Pipelines (StandardScaler + Estimador):
     - Random Forest (`n_estimators=100, max_depth=10`)
     - SVM (`kernel=rbf, C=1.0`)
     - Gradient Boosting (`n_estimators=100, max_depth=5, lr=0.1`)
   - **Evaluación**: Accuracy, precision, recall, F1 (weighted) + matriz de confusión.
   - **Persistencia**: Cada pipeline se guarda como `{nombre}_model.pkl` en el volumen compartido `/app/models/`. Las métricas se guardan en `/app/report/model_metrics.pkl`.

4. Una vez completado, la API detecta automáticamente los modelos disponibles.

### Agregar un nuevo modelo

Para entrenar un modelo adicional, agregar una celda en el notebook:

```python
from sklearn.neighbors import KNeighborsClassifier

knn_metrics = trainer.train_and_save(
    name='knn',
    estimator=KNeighborsClassifier(n_neighbors=5),
    X_train=X_train, X_test=X_test,
    y_train=y_train, y_test=y_test,
    scaler=StandardScaler(),
)
```

La API lo detectará automáticamente en el endpoint `GET /models`.

## Pruebas de la API

### Listar modelos disponibles

```bash
curl http://localhost:8000/models
```

Respuesta esperada (después de entrenar):

```json
{
  "available_models": [
    {
      "name": "gradientboosting",
      "endpoint": "POST /classify/gradientboosting",
      "metrics": { "train_accuracy": 1.0, "test_accuracy": 0.98, ... }
    },
    {
      "name": "randomforest",
      "endpoint": "POST /classify/randomforest",
      "metrics": { ... }
    },
    {
      "name": "svm",
      "endpoint": "POST /classify/svm",
      "metrics": { ... }
    }
  ]
}
```

### Clasificar un pingüino

```bash
curl -X POST http://localhost:8000/classify/randomforest \
  -H "Content-Type: application/json" \
  -d '{
    "island": 1,
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750,
    "sex": 1,
    "year": 2007
  }'
```

Respuesta:

```json
{
  "model": "randomforest",
  "species_id": 1,
  "species_name": "Adelie"
}
```

### Documentación interactiva

FastAPI genera documentación Swagger automáticamente en `http://localhost:8000/docs`.

## Registro de Resultados (Logging de Predicciones)

La API registra automáticamente cada predicción en un archivo de log persistente. Este mecanismo permite auditar las predicciones realizadas y analizar el uso de los modelos.

### Cómo funciona

Al iniciar la API, se configura un logger dedicado usando el módulo `logging` de Python:

```python
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

_pred_logger = logging.getLogger("predictions")
_pred_logger.setLevel(logging.INFO)
_handler = logging.FileHandler(os.path.join(RESULTS_DIR, "predictions.log"))
_handler.setFormatter(logging.Formatter("%(message)s"))
_pred_logger.addHandler(_handler)
```

- Se crea el directorio `results/` si no existe.
- Se configura un logger llamado `"predictions"` con un `FileHandler` que escribe en `results/predictions.log`.
- El formato es solo `%(message)s` (sin prefijos de nivel o timestamp del logger, porque el timestamp se incluye manualmente en el JSON).

### Qué se registra

Cada vez que el endpoint `POST /classify/{model_name}` responde exitosamente, se escribe una línea JSON en el log con tres campos:

```python
log_entry = {
    "timestamp": datetime.utcnow().isoformat(),   # cuándo se hizo la predicción
    "input": data.model_dump(),                     # los 7 campos de entrada del pingüino
    "result": {
        "model": model_name,                        # qué modelo se usó
        "species_id": prediction,                   # predicción numérica (1, 2 o 3)
        "species_name": species_name,               # nombre de la especie
    },
}
_pred_logger.info(json.dumps(log_entry))
```

### Ejemplo de una línea en predictions.log

```json
{
  "timestamp": "2025-01-15T14:32:01.123456",
  "input": {
    "island": 1,
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750,
    "sex": 1,
    "year": 2007
  },
  "result": {
    "model": "randomforest",
    "species_id": 1,
    "species_name": "Adelie"
  }
}
```

### Persistencia del log

El archivo `predictions.log` se almacena en el volumen Docker `shared_results`, montado en `/app/results/` dentro del contenedor de la API. Esto significa que:

- El log sobrevive a reinicios del contenedor (`docker-compose down` + `up`).
- Se pierde solo si se eliminan los volúmenes explícitamente (`docker-compose down -v`).
- Cada predicción se escribe inmediatamente al archivo (no hay buffer), lo que garantiza que no se pierden registros ante un crash del contenedor.

## Mapeo de Variables

| Campo | Tipo | Rango / Valores |
|---|---|---|
| island | int | 1, 2, 3 |
| bill_length_mm | float | 10.0 – 100.0 |
| bill_depth_mm | float | 5.0 – 35.0 |
| flipper_length_mm | int | 100 – 300 |
| body_mass_g | int | 1000 – 10000 |
| sex | int | 0 (hembra), 1 (macho) |
| year | int | 2000 – 2030 |

Especies: `1` = Adelie, `2` = Chinstrap, `3` = Gentoo

## Volúmenes Compartidos

| Volumen | Jupyter | API | Contenido |
|---|---|---|---|
| shared_models | /app/models | /app/modelos | Archivos .pkl de modelos |
| shared_report | /app/report | /app/report | model_metrics.pkl |
| shared_data | /app/data | — | penguins_v1.csv |
| shared_results | /app/results | /app/results | predictions.log |

## Notas

- La API carga modelos dinámicamente: cualquier archivo `*_model.pkl` en el volumen de modelos es detectado sin reiniciar el servicio.
- Las predicciones se registran en `results/predictions.log` con timestamp, input y resultado.
- Si la API se levanta sin modelos entrenados, `GET /models` retorna una lista vacía y `POST /classify/{model}` retorna 404.
