import os
import glob

import joblib
import numpy as np
import pandas as pd


MODELS_DIR = "models"
REPORT_PATH = "report/model_metrics.pkl"
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


def is_pipeline(model):
    """Detecta si el modelo cargado es un Pipeline de sklearn"""
    return hasattr(model, "steps")
