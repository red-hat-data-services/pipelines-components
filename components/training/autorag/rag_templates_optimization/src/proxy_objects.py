"""
This module contains proxy classes for respective classes from `ai4rag` module.
The proxies defined here exist so to ease the local execution, debugging or
unit/integration-testing by allowing mocked runs of `ai4rag` code without an
external llama-stack server setup.
"""

from typing import Sequence

from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.experiment.mps import ModelsPreSelector
from ai4rag.core.experiment.results import EvaluationData, EvaluationResult, ExperimentResults
from ai4rag.utils.event_handler.event_handler import BaseEventHandler, LogLevel


class StdoutEventHandler(BaseEventHandler):

    def __init__(self, event_handler: BaseEventHandler) -> None:
        self.event_handler = event_handler
        super().__init__()

    def on_status_change(self, level: LogLevel, message: str, step: str | None = None) -> None:
        pass

    def on_pattern_creation(self, payload: dict, evaluation_results: list, **kwargs) -> None:
        pass


class DisconnectedAI4RAGExperiment(AI4RAGExperiment):

    def __init__(self, rag_experiment: AI4RAGExperiment) -> None:
        self.rag_experiment = rag_experiment
        self.metrics = ["faithfulness"]
        # self.metrics = rag_experiment.metrics

    def search(self, **kwargs) -> Sequence[EvaluationResult]:

        # set mocked self.results and return

        self.results = ExperimentResults()

        for i in range(3):
            eval_res = EvaluationResult(
                f"pattern{i}",
                f"collection{i}",
                {"indexing_param_key": f"indexing_val{i}"},
                {"rag_param_key": f"rag_param_val{i}"},
                scores={
                    "scores": {
                        "answer_correctness": {"mean": 0.1 * i, "ci_low": 0.4, "ci_high": 0.6},
                        "faithfulness": {"mean": 0.1 * i, "ci_low": 0.4, "ci_high": 0.6},
                        "context_correctness": {"mean": 1.0, "ci_low": 0.9, "ci_high": 1.0},
                    },
                    "question_scores": {
                        "answer_correctness": {"q_id_0": 0.0, "q_id_1": 0.0, "q_id_2": 0.0146},
                        "faithfulness": {"q_id_0": 0.0909, "q_id_1": 0.1818, "q_id_2": 0.1818},
                        "context_correctness": {"q_id_0": 0.0, "q_id_1": 0.2, "q_id_2": 0.0},
                    },
                },
                execution_time=0.5 * i,
                final_score=0.1 * i,
            )
            eval_data = [
                EvaluationData(
                    question="What foundation models are available in watsonx.ai?",
                    answer="I cannot answer this question, because I am just a mocked model.",
                    contexts=[
                        "*  asset_name_or_item: (Required) Either a string with the name of a stored data asset or an item like those returned by list_stored_data().",
                        "Model architecture   The architecture of the model influences how the model behaves.",
                        "Learn more \n\nParent topic:[Governing assets in AI use cases]",
                    ],
                    context_ids=[
                        "0ECEAC44DA213D067B5B5EA66694E6283457A441_9.txt",
                        "120CAE8361AE4E0B6FE4D6F0D32EEE9517F11190_1.txt",
                        "391DBD504569F02CCC48B181E3B953198C8F3C8A_8.txt",
                    ],
                    ground_truths=[
                        "The following models are available in watsonx.ai: \nflan-t5-xl-3b\nFlan-t5-xxl-11b\nflan-ul2-20b\ngpt-neox-20b\ngranite-13b-chat-v2\ngranite-13b-chat-v1\ngranite-13b-instruct-v2\ngranite-13b-instruct-v1\nllama-2-13b-chat\nllama-2-70b-chat\nmpt-7b-instruct2\nmt0-xxl-13b\nstarcoder-15.5b",
                    ],
                    question_id="q_id_0",
                    ground_truths_context_ids=None,
                ),
                EvaluationData(
                    question="What foundation models are available on Watsonx, and which of these has IBM built?",
                    answer="I cannot answer this question, because I am just a mocked model.",
                    contexts=[
                        "Retrieval-augmented generation \n\nYou can use foundation models in IBM watsonx.ai to generate factually accurate output.",
                        "Methods for tuning foundation models \n\nLearn more about different tuning methods and how they work.",
                        "Foundation models built by IBM \n\nIn IBM watsonx.ai, you can use IBM foundation models that are built with integrity and designed for business.",
                    ],
                    context_ids=[
                        "752D982C2F694FFEE2A312CEA6ADF22C2384D4B2_0.txt",
                        "15A014C514B00FF78C689585F393E21BAE922DB2_0.txt",
                        "B2593108FA446C4B4B0EF5ADC2CD5D9585B0B63C_0.txt",
                    ],
                    ground_truths=[
                        "The following foundation models are available on Watsonx:\n\n1. flan-t5-xl-3b\n2. flan-t5-xxl-11b\n3. flan-ul2-20b\n4. gpt-neox-20b\n5. granite-13b-chat-v2 (IBM built)\n6. granite-13b-chat-v1 (IBM built)\n7. granite-13b-instruct-v2 (IBM built)\n8. granite-13b-instruct-v1 (IBM built)\n9. llama-2-13b-chat\n10. llama-2-70b-chat\n11. mpt-7b-instruct2\n12. mt0-xxl-13b\n13. starcoder-15.5b\n\n The Granite family of foundation models, including granite-13b-chat-v2, granite-13b-chat-v1, and granite-13b-instruct-v2 has been build by IBM.",
                    ],
                    question_id="q_id_1",
                    ground_truths_context_ids=None,
                ),
                EvaluationData(
                    question="How can I ensure that the generated answers will be accurate, factual and based on my information?",
                    answer="I cannot answer this question, because I am just a mocked model.",
                    contexts=[
                        "Functions used in Watson Pipelines's Expression Builder \n\nUse these functions in Pipelines code editors.",
                        "Table 1. Supported values, defaults, and usage notes for sampling decoding\n\n Parameter        Supported values                                                                                 Default  Use",
                        "applygmm properties \n\nYou can use the Gaussian Mixture node to generate a Gaussian Mixture model nugget.",
                    ],
                    context_ids=[
                        "E933C12C1DF97E13CBA40BCD54E4F4B8133DA10C_0.txt",
                        "42AE491240EF740E6A8C5CF32B817E606F554E49_1.txt",
                        "F2D3C76D5EABBBF72A0314F29374527C8339591A_0.txt",
                    ],
                    ground_truths=[
                        "To ensure a language model provides the most accurate and factual answers to questions based on your data, you can follow these steps:\n1. Utilize Retrieval-augmented generation pattern. In this pattaern, you provide the relevant facts from your dataset as context in your prompt text. This will guide the model to generate responses grounded in the provided data\n2. Prompt Engineering: Experiment with prompt engineering techniques to shape the model's output. Understand the capabilities and limitations of the foundation model by fine-tuning prompts and adjusting inputs to align with the desired output. This process helps in refining the generated responses for accuracy.\n3. Review and Validate Output: Regularly review the generated output for biased, inappropriate, or incorrect content. Third-party models may produce outputs containing misinformation, offensive language, or biased content. Implement mechanisms to evaluate and validate the accuracy of the model's responses, ensuring alignment with factual information from your dataset.\n",
                    ],
                    question_id="q_id_2",
                    ground_truths_context_ids=None,
                ),
            ]

            self.results.add_evaluation(eval_data, eval_res)

        self.search_output = self.results.get_best_evaluations(k=1)

        return self.search_output


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
