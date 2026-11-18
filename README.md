# EPHPM Honduras: indicadores por departamento y año

Procesa las bases de la Encuesta Permanente de Hogares de Propósitos Múltiples (INE Honduras), arma un panel de indicadores por departamento y año, entrena un modelo que asigna un nivel de condiciones de vida y lo sirve con una API FastAPI. El dashboard vive en el repo `frontend-EPHPM-2025`.

## Datos

Fuente: <https://ine.gob.hn/bases-de-datos/>. Las bases no se versionan; se descargan a `data/raw/<año>/` (ver `data/raw/MANIFEST.md`).

| Cortes en el panel | Notas |
|---|---|
| 2007–2011, 2018, 2019, 2023–2025 | Todos los indicadores, incluida pobreza |
| 2021, 2022 | Sin pobreza; el ingreso del hogar se deriva sumando el de sus miembros |
| 2001–2006 | Fuera: no traen ingreso del hogar comparable |
| 2012–2017, 2020 | El INE no publica microdatos descargables |

Particularidades que el ETL resuelve (`config/variables.yml`):

- En 2019, 2022 y 2023 el departamento se toma del código de segmento (`COR_PRE // 10000`), porque la columna de departamento falta o viene casi vacía.
- La base 2018 etiqueta al revés los códigos 9 y 11 (Gracias a Dios e Islas de la Bahía).
- Las bases 2023 y 2024 registran 9 años de estudio a quien tiene educación media; por eso la educación se usa solo en tres tramos (menos de 6, 6 a 12, 13 o más).
- Todas las cifras se ponderan con el factor de expansión.

Indicadores que no son comparables entre años y no se publican: NBI, años de estudio promedio y cuenta propia.

## Pipeline

```bash
pip install -r requirements.txt -r requirements_local.txt
python src/inventory.py                          # variables disponibles en cada base
python src/etl.py                                # data/processed/{personas,hogares,depto_anio}.parquet
python src/clustering.py evaluate                # compara variables, algoritmos y k
python src/clustering.py fit --set nucleo --k 3  # models/clustering_panel.onnx y .json
python src/export_panel.py --front "../EPHPM Front/frontend-EPHPM-2025"
python src/poverty_model.py                      # baseline supervisado de pobreza por hogar
PYTHONPATH=src python src/poverty_logreg.py      # regresión logística de pobreza: ajuste, prueba y ONNX
```

## Modelo de niveles

`StandardScaler + KMeans(k=3)` sobre 164 observaciones departamento-año y 6 indicadores: pobreza, pobreza extrema, ingreso mediano relativo al país, ocupación, desempleo y dependencia demográfica. Los clusters se numeran de mejor (0) a peor (2) por pobreza media. El silhouette es 0.27: son tramos de un gradiente, no grupos nítidos.

El ONNX incluye el escalado; la entrada se pasa sin escalar.

## API

Los tres artefactos se cargan en memoria desde Vercel Blob al arrancar:

| Variable | Archivo |
|---|---|
| `MODEL_BLOB_URL` | `models/clustering_panel.onnx` |
| `MODEL_META_BLOB_URL` | `models/clustering_panel.json` |
| `PANEL_BLOB_URL` | `api_data/panel.json` |

Si falta alguna, la API usa esos archivos locales (solo desarrollo; no están en git). Tras reentrenar o regenerar el panel hay que subir los tres archivos al Blob.

```bash
uvicorn main_onnx:app --reload
```

| Endpoint | Devuelve |
|---|---|
| `GET /years` | Cortes disponibles y cuáles tienen nivel |
| `GET /indicators` | Definición de cada indicador |
| `GET /national` | Serie nacional |
| `GET /departments?year=` | Indicadores y nivel de cada departamento en un año |
| `GET /departments/{code}/series` | Serie de un departamento (código 1–18) |
| `GET /clusters` | Los tres niveles y su perfil |
| `POST /predict` | Nivel para 6 indicadores |
| `POST /predict-batch` | Lo mismo para una lista |

## Modelo de pobreza por hogar

Regresión logística que estima la probabilidad de que un hogar sea pobre sin usar ingresos (la pobreza se define con el ingreso). Variables: tamaño y composición del hogar, ocupados por persona, edad, sexo, ocupación y educación del jefe, educación máxima del hogar, área y departamento.

| Paso | Años |
|---|---|
| Entrenamiento | 2007–2011, 2018, 2019 |
| Validación (regularización y umbral) | 2023, 2024 |
| Prueba, una sola vez | 2025 |

Resultado en 2025: ROC-AUC 0.785, exactitud balanceada 0.711, con umbral 0.50. LightGBM rinde igual en validación (0.784 frente a 0.781), así que se queda el modelo lineal.

Limitación conocida: en 2025 el modelo subestima la pobreza entre 4 y 13 puntos en todos los tramos de probabilidad. Los hogares mejoran en educación y tamaño más rápido de lo que baja la pobreza, así que un modelo entrenado con años anteriores queda optimista. El modelo exportado se reajusta con todos los años, pero conviene recalibrarlo con cada encuesta nueva.

```bash
python src/poverty_model.py                    # comparación de modelos base
PYTHONPATH=src python src/poverty_logreg.py    # ajuste, prueba y models/poverty_logreg.onnx
```

El ONNX recibe una entrada por variable (ver `models/poverty_logreg.json`) y, como los demás artefactos, no se versiona en git.
