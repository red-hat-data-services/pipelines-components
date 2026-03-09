"""
This module contains proxy classes for respective classes from `ai4rag` module.
The proxies defined here exist so to ease the local execution, debugging or
unit/integration-testing by allowing mocked runs of `ai4rag` code without an
external llama-stack server setup.
"""

from ai4rag.core.experiment.mps import ModelsPreSelector
from ai4rag.rag.embedding.base_model import EmbeddingModel
from ai4rag.rag.foundation_models.base_model import FoundationModel


class DisconnectedModelsPreSelector(ModelsPreSelector):

    def __init__(self, mps: ModelsPreSelector) -> None:
        self.mps: ModelsPreSelector = mps
        self.metric = mps.metric

    def evaluate_patterns(self):

        self.evaluation_results = [
            {
                "embedding_model": "granite_emb1",
                "foundation_model": "mistral1",
                "scores": {"faithfulness": {"mean": 0.5, "ci_low": 0.4, "ci_high": 0.6}},
                "question_scores": {
                    "faithfulness": {
                        "q_id_0": 0.5,
                        "q_id_1": 0.8,
                    }
                },
            },
            {
                "embedding_model": "granite_emb2",
                "foundation_model": "mistral2",
                "scores": {"faithfulness": {"mean": 0.5, "ci_low": 0.4, "ci_high": 0.6}},
                "question_scores": {
                    "faithfulness": {
                        "q_id_0": 0.5,
                        "q_id_1": 0.8,
                    }
                },
            },
        ]

    # def select_models(self, n_em: int = 2, n_fm: int = 3) -> dict[str, list[EmbeddingModel | FoundationModel]]:
    #     pass
