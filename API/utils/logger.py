import os
import json
import logging
from datetime import datetime


class PredictionLogger:
    """Logger dedicado para registrar predicciones en archivo."""

    def __init__(self, results_dir="results", filename="predictions.log"):
        os.makedirs(results_dir, exist_ok=True)
        self._logger = logging.getLogger("predictions")
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            handler = logging.FileHandler(os.path.join(results_dir, filename))
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    def log(self, input_data: dict, result: dict):
        """Registra una predicción con timestamp, entrada y resultado."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "input": input_data,
            "result": result,
        }
        self._logger.info(json.dumps(entry))
