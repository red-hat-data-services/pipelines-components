"""Proxy classes for ai4rag for local execution and testing without llama-stack."""

from ai4rag.core.experiment.mps import ModelsPreSelector


class DisconnectedModelsPreSelector(ModelsPreSelector):
    """ModelsPreSelector that returns mocked evaluation results."""

    def __init__(self, mps: ModelsPreSelector) -> None:
        """Wrap the given ModelsPreSelector; use its metric."""
        self.mps: ModelsPreSelector = mps
        self.metric = mps.metric

    def evaluate_patterns(self):
        """Set evaluation_results to mocked data."""
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
