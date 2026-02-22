"""
Clase encargada de entrenar modelos, guardarlos, mostrar métricas visuales y actualizar el reporte
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


class ModelTrainer:
    """Entrena pipelines de sklearn, los persiste y mantiene el reporte de métricas"""

    def __init__(
        self,
        models_dir: str = "/app/models",
        report_path: str = "/app/report/model_metrics.pkl",
    ):
        self.models_dir = models_dir
        self.report_path = report_path
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

    def train_and_save(self, name: str, estimator, X_train, X_test, y_train, y_test, scaler=None):
        """
        Construye un Pipeline, entrena, evalúa, muestra classification_report y matriz de confusión, guarda el pipeline como {name}_model.pkl y actualiza el reporte.

        Parámetros
        ----------
        name : str          Nombre del modelo (ej. 'randomforest').
        estimator :         Estimador de sklearn (sin entrenar).
        X_train, X_test :   Features (sin escalar si se pasa scaler).
        y_train, y_test :   Labels.
        scaler :            Transformador opcional (ej. StandardScaler()).
                            Si se pasa, se integra en el Pipeline.

        Retorna
        -------
        dict con las métricas calculadas.
        """
        pipeline = self._build_pipeline(estimator, scaler)
        pipeline.fit(X_train, y_train)

        metrics = self._evaluate(pipeline, name, X_train, X_test, y_train, y_test)
        self._show_report(pipeline, name, X_test, y_test)

        # Guardar pipeline completo
        model_path = os.path.join(self.models_dir, f"{name.lower()}_model.pkl")
        joblib.dump(pipeline, model_path)

        # Actualizar reporte
        self._update_report(metrics)

        print(f"\nPipeline '{name}' guardado en {model_path}")
        return metrics

    def _build_pipeline(self, estimator, scaler=None) -> Pipeline:
        steps = []
        if scaler is not None:
            steps.append(("scaler", scaler))
        steps.append(("model", estimator))
        return Pipeline(steps)

    def _evaluate(self, pipeline, name, X_train, X_test, y_train, y_test) -> dict:
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        return {
            "model": name.lower(),
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred, average="weighted"),
            "test_recall": recall_score(y_test, y_test_pred, average="weighted"),
            "test_f1": f1_score(y_test, y_test_pred, average="weighted"),
        }

    def _show_report(self, pipeline, name, X_test, y_test):
        y_pred = pipeline.predict(X_test)

        # Classification report
        print(f"\n{'='*50}")
        print(f"  {name} — Classification Report")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred))

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusión — {name}")
        plt.ylabel("Valor Real")
        plt.xlabel("Predicción")
        plt.tight_layout()
        plt.show()

    def _update_report(self, metrics: dict):
        if os.path.exists(self.report_path):
            df = pd.read_pickle(self.report_path)
        else:
            df = pd.DataFrame()

        model_name = metrics["model"]
        if not df.empty and "model" in df.columns:
            df = df[df["model"] != model_name]

        df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
        df.to_pickle(self.report_path)
