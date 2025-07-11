import pandas as pd
import mlflow
import faulthandler
from typing import Any
import os

faulthandler.enable()


class MLflowLoader:
    def __init__(self, tracking_uri: str, experiment_name: str):
        """MLflow 서버 URI와 실험 이름을 받아 초기화합니다."""
        self.tracking_uri: str = tracking_uri
        self.experiment_name: str = experiment_name

        mlflow.set_tracking_uri(uri=self.tracking_uri)
        self.experiment_id: int = mlflow.get_experiment_by_name(
            name=self.experiment_name
        ).experiment_id

    def get_latest_runs(self) -> dict:
        """
        지정된 실험(experiment)에서 가장 최근의 모델과 실행(Run)을
        각각 하나씩 찾아 반환해야 합니다.
        - 반환 형태: {'SVD': MLflow_Run_Object, 'NCF': MLflow_Run_Object}
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
        )
        latest_runs = (
            runs.sort_values("start_time", ascending=False)
            .groupby("tags.mlflow.runName", as_index=False)
            .first()
        )
        return {
            model: run_obj
            for model, run_obj in zip(
                latest_runs["tags.mlflow.runName"], latest_runs.iterrows()
            )
        }

    def load_models(self, runs: dict):
        """
        get_latest_runs에서 반환된 Run 객체 딕셔너리를 받아,
        각 Run에 저장된 모델 아티팩트를 불러옵니다.
        - 반환 형태: {'SVD': Loaded_SVD_Model, 'NCF': Loaded_NCF_Model}
        """
        loaded_models = {}
        CACHE_DIR = "./mlartifacts"

        for model_name, (_, run_obj) in runs.items():
            model_uri = f"runs:/{run_obj.run_id}/{model_name}"
            print(f"Loading model for {model_name} from: {model_uri}")
            loaded_models[model_name] = mlflow.pyfunc.load_model(model_uri)

        return loaded_models

    def get_run_metrics_and_params(self, runs: dict) -> dict[str, dict[str, Any]]:
        """
        Run 객체들로부터 주요 평가지표(metrics)와 하이퍼파라미터(params)를
        추출하여 정리된 형태로 반환합니다.
        - 목적: 비용-효과 분석에 사용할 학습 시간, 오프라인 성능 등을 가져오기 위함입니다.
        - 반환 형태: {'SVD': {'metrics': {...}, 'params': {...}, 'training_time': ...},
                     'NCF': {'metrics': {...}, 'params': {...}, 'training_time': ...}}
        """
        metrics_and_params = {}
        client = mlflow.tracking.MlflowClient()

        for model_name, (_, run_obj) in runs.items():
            run_id = run_obj.run_id
            run = client.get_run(run_id)

            # Get metrics, including the explicitly logged training_time
            metrics = run.data.metrics
            params = run.data.params

            # Use the logged training_time if available, otherwise calculate it.
            training_time = metrics.get("training_time")
            if training_time is None:
                training_time = (
                    run.info.end_time - run.info.start_time
                ) / 1000.0  # Convert ms to seconds

            metrics_and_params[model_name] = {
                "metrics": metrics,
                "params": params,
                "training_time": training_time,
            }

        return metrics_and_params
