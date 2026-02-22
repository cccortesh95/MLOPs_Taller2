import os
import glob
import json
import logging
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="Penguin Classifier API")

MODELS_DIR = "models"
REPORT_PATH = "report/model_metrics.pkl"
RESULTS_DIR = "results"

# Configurar logger de predicciones
os.makedirs(RESULTS_DIR, exist_ok=True)
_pred_logger = logging.getLogger("predictions")
_pred_logger.setLevel(logging.INFO)
_handler = logging.FileHandler(os.path.join(RESULTS_DIR, "predictions.log"))
_handler.setFormatter(logging.Formatter("%(message)s"))
_pred_logger.addHandler(_handler)

SPECIES_MAP = {1: "Adelie", 2: "Chinstrap", 3: "Gentoo"}


def load_metrics():
    """Carga métricas si el archivo existe."""
    if os.path.exists(REPORT_PATH):
        df = pd.read_pickle(REPORT_PATH)
        return {
            row["model"].lower(): {
                "train_accuracy": row["train_accuracy"],
                "test_accuracy": row["test_accuracy"],
                "test_precision": row["test_precision"],
                "test_recall": row["test_recall"],
                "test_f1": row["test_f1"],
            }
            for _, row in df.iterrows()
        }
    return {}


def discover_models():
    """Descubre dinámicamente los modelos .pkl disponibles en el directorio."""
    models = {}
    pattern = os.path.join(MODELS_DIR, "*_model.pkl")
    for path in glob.glob(pattern):
        filename = os.path.basename(path)
        name = filename.replace("_model.pkl", "").lower()
        models[name] = path
    return models


def load_model(path):
    """Carga un modelo desde disco"""
    return joblib.load(path)


def load_scaler():
    """Carga el scaler si existe"""
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None


def _is_pipeline(model):
    """Detecta si el modelo cargado es un Pipeline de sklearn."""
    return hasattr(model, "steps")


class PenguinInput(BaseModel):
    island: int = Field(default=1, examples=[1])
    bill_length_mm: float = Field(default=39.1, examples=[39.1])
    bill_depth_mm: float = Field(default=18.7, examples=[18.7])
    flipper_length_mm: int = Field(default=181, examples=[181])
    body_mass_g: int = Field(default=3750, examples=[3750])
    sex: int = Field(default=1, examples=[1])
    year: int = Field(default=2007, examples=[2007])

    @field_validator("island")
    def validate_island(cls, v):
        if v not in (1, 2, 3):
            raise ValueError("island debe ser 1, 2 o 3")
        return v

    @field_validator("bill_length_mm")
    def validate_bill_length(cls, v):
        if not (10.0 <= v <= 100.0):
            raise ValueError("bill_length_mm debe estar entre 10.0 y 100.0")
        return v

    @field_validator("bill_depth_mm")
    def validate_bill_depth(cls, v):
        if not (5.0 <= v <= 35.0):
            raise ValueError("bill_depth_mm debe estar entre 5.0 y 35.0")
        return v

    @field_validator("flipper_length_mm")
    def validate_flipper_length(cls, v):
        if not (100 <= v <= 300):
            raise ValueError("flipper_length_mm debe estar entre 100 y 300")
        return v

    @field_validator("body_mass_g")
    def validate_body_mass(cls, v):
        if not (1000 <= v <= 10000):
            raise ValueError("body_mass_g debe estar entre 1000 y 10000")
        return v

    @field_validator("sex")
    def validate_sex(cls, v):
        if v not in (0, 1):
            raise ValueError("sex debe ser 0 o 1")
        return v

    @field_validator("year")
    def validate_year(cls, v):
        if not (2000 <= v <= 2030):
            raise ValueError("year debe estar entre 2000 y 2030")
        return v


def _build_features(data: PenguinInput) -> np.ndarray:
    bill_ratio = data.bill_length_mm / data.bill_depth_mm
    body_mass_kg = data.body_mass_g / 1000
    return np.array([[
        data.island, data.bill_length_mm, data.bill_depth_mm,
        data.flipper_length_mm, data.body_mass_g, data.sex,
        data.year, bill_ratio, body_mass_kg
    ]])


@app.get("/models")
async def list_models():
    """Lista todos los modelos disponibles dinámicamente."""
    available = discover_models()
    metrics = load_metrics()
    return {
        "available_models": [
            {
                "name": name,
                "endpoint": f"POST /classify/{name}",
                "metrics": metrics.get(name, {}),
            }
            for name in sorted(available.keys())
        ]
    }


@app.post("/classify/{model_name}")
async def classify(model_name: str, data: PenguinInput):
    """Clasifica un pingüino usando el modelo especificado (carga dinámica)."""
    available = discover_models()
    if model_name not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{model_name}' no encontrado. Usa GET /models para ver los disponibles.",
        )
    try:
        model = load_model(available[model_name])
        features = _build_features(data)
        if _is_pipeline(model):
            # Pipeline ya incluye scaler + modelo
            prediction = int(model.predict(features)[0])
        else:
            # Modelo suelto: aplicar scaler externo si existe
            scaler = load_scaler()
            if scaler is not None:
                features = scaler.transform(features)
            prediction = int(model.predict(features)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    species_name = SPECIES_MAP.get(prediction)
    if species_name is None:
        raise HTTPException(status_code=404, detail="Especie no encontrada")

    result = {
        "model": model_name,
        "species_id": prediction,
        "species_name": species_name,
    }

    # Registrar entrada, modelo y resultado en el log
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": data.model_dump(),
        "result": result,
    }
    _pred_logger.info(json.dumps(log_entry))

    return result
