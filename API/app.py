import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from utils.model_utils import (
    SPECIES_MAP,
    discover_models,
    is_pipeline,
    load_metrics,
    load_model,
    load_scaler,
)
from utils.logger import PredictionLogger

app = FastAPI(title="Penguin Classifier API")
pred_logger = PredictionLogger()


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
    """Clasifica un pingüino usando el modelo especificado"""
    available = discover_models()
    if model_name not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{model_name}' no encontrado. Usa GET /models para ver los disponibles.",
        )
    try:
        model = load_model(available[model_name])
        features = _build_features(data)
        if is_pipeline(model):
            prediction = int(model.predict(features)[0])
        else:
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

    pred_logger.log(data.model_dump(), result)

    return result
