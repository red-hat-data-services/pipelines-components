from typing import Optional

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",
    packages_to_install=[
        "ai4rag@git+https://github.com/IBM/ai4rag.git",
        "pyyaml",
        "pysqlite3-binary",  # ChromaDB requires sqlite3 >= 3.35; base image has older sqlite
        "llama-stack-client",
        "openai",
    ],
)
def rag_templates_optimization(
    extracted_text: dsl.InputPath(dsl.Artifact),
    test_data: dsl.InputPath(dsl.Artifact),
    search_space_prep_report: dsl.InputPath(dsl.Artifact),
    rag_patterns: dsl.Output[dsl.Artifact],
    autorag_run_artifact: dsl.Output[dsl.Artifact],
    chat_model_url: Optional[str] = None,
    chat_model_token: Optional[str] = None,
    embedding_model_url: Optional[str] = None,
    embedding_model_token: Optional[str] = None,
    llama_stack_vector_database_id: Optional[str] = None,
    optimization_settings: Optional[dict] = None,
    input_data_key: Optional[str] = None,
    test_data_key: Optional[str] = None,
):
    """RAG Templates Optimization component.

    Carries out the iterative RAG optimization process.

    Args:
        extracted_text: A path pointing to a folder containg extracted texts from input documents.

        test_data: A path pointing to test data used for evaluating RAG pattern quality.

        search_space_prep_report: A path pointing to a .yml file containig short
            report on the experiment's first phase (search space preparation).

        rag_patterns: kfp-enforced argument specifying an output artifact. Provided by kfp backend automatically.

        autorag_run_artifact: kfp-enforced argument specifying an output artifact. Provided by kfp backend atomatically.

        chat_model_url: Inference endpoint URL for the chat/generation model (OpenAI-compatible).
            Required for in-memory scenario.

        chat_model_token: Optional API token for the chat model endpoint. Omit if deployment has no auth.

        embedding_model_url: Inference endpoint URL for the embedding model. Required for in-memory scenario.

        embedding_model_token: Optional API token for the embedding model endpoint. Omit if no auth.

        vector_database: An identificator of the vector store used in the experiment.

        llama_stack_vector_database_id: Vector database identifier as registered in llama-stack.

        optimization_settings: Additional settings customising the experiment.

        input_data_key: A path to documents dir within a bucket used as an input to AI4RAG experiment.
        test_data_key: A path to test data file within a bucket used as an input to AI4RAG experiment.

    Returns:
        rag_patterns: Folder containing all generated RAG patterns (each subdir: pattern.json,
            indexing_notebook.ipynb, inference_notebook.ipynb).
        autorag_run_artifact: Run log and experiment status (TODO).
    """
    # ChromaDB (via ai4rag) requires sqlite3 >= 3.35; RHEL9 base image has older sqlite.
    # Patch stdlib sqlite3 with pysqlite3-binary before any ai4rag import.
    import sys

    try:
        import pysqlite3

        sys.modules["sqlite3"] = pysqlite3
    except ImportError:
        pass

    import os
    from collections import namedtuple
    from json import dump as json_dump
    from pathlib import Path
    from typing import Any, Literal, Self
    from string import Formatter
    from copy import deepcopy

    import pandas as pd
    import yaml as yml
    from ai4rag.core.experiment.experiment import AI4RAGExperiment
    from ai4rag.core.experiment.results import ExperimentResults
    from ai4rag.core.hpo.gam_opt import GAMOptSettings
    from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
    from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel
    from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
    from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
    from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
    from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
    from ai4rag.search_space.src.parameter import Parameter
    from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
    from ai4rag.utils.event_handler.event_handler import BaseEventHandler, LogLevel
    from langchain_core.documents import Document
    from llama_stack_client import LlamaStackClient
    from openai import OpenAI

    MAX_NUMBER_OF_RAG_PATTERNS = 8
    METRIC = "faithfulness"
    SUPPORTED_OPTIMIZATION_METRICS = frozenset({"faithfulness", "answer_correctness", "context_correctness"})

    class NotebookCell:
        """
        Represents a single cell in a Jupyter notebook.

        Parameters
        ----------
        cell_type : Literal["code", "markdown"]
            The type of cell.
        source : str | list[str]
            The cell content. Can be a string or list of strings.
        metadata : dict, optional
            Cell metadata.
        """

        def __init__(
            self,
            cell_type: Literal["code", "markdown"],
            source: str | list[str],
            metadata: dict | None = None,
        ):
            self.cell_type = cell_type
            self.metadata = metadata or {}

            self.source = source

            if cell_type == "code":
                self.execution_count = None
                self.outputs = []

        def to_dict(self) -> dict:
            """
            Convert cell to notebook JSON format.

            Returns
            -------
            dict
                Cell in notebook format.
            """
            cell_dict = {
                "cell_type": self.cell_type,
                "metadata": self.metadata,
                "source": self.source,
            }

            if self.cell_type == "code":
                cell_dict["execution_count"] = self.execution_count
                cell_dict["outputs"] = self.outputs

            return cell_dict

        def format_source(
            self,
            placeholders_mapping: dict,
        ) -> Self:
            """
            Formats cell source based on provided placeholders_mapping.

            Returns
            -------
            Self
                Instance of NotebookCell.
            """
            if isinstance(self.source, list):
                new_source = []
                for line in self.source:
                    line_mapping = {}
                    for _, field_name, _, _ in Formatter().parse(line):
                        if field_name is None:
                            continue
                        line_mapping[field_name] = placeholders_mapping.get(field_name, "")

                    new_source.append(line.format(**line_mapping))
                    self.source = new_source

                return self

            self.source = self.source.format(**placeholders_mapping)

            return self

    class Notebook:
        """
        Builder class for creating and manipulating Jupyter notebooks.

        This class provides a fluent API for programmatically building notebooks
        by adding code and markdown cells, formatting content, and saving to disk.

        Parameters
        ----------
        kernel_name : str, default="python3"
            Kernel name for the notebook.
        kernel_display_name : str, default="Python 3"
            Display name for the kernel.
        language : str, default="python"
            Programming language.
        language_version : str, default="3.11.0"
            Language version.
        cells : list[NotebookCell] | None, default=None
            Notebook cells to build the notebook from.

        Examples
        --------
        >>> nb = Notebook(
            cells=[
                NotebookCell(
                    cell_type="markdown",
                    source="### Hello world!",
                )
            ])
        >>> nb.save("output.ipynb")
        """

        def __init__(
            self,
            kernel_name: str = "python3",
            kernel_display_name: str = "Python 3",
            language: str = "python",
            language_version: str = "3.13.11",
            cells: list[NotebookCell] | None = None,
        ):
            self.cells: list[NotebookCell] = cells if cells else []
            self.metadata = {
                "kernelspec": {
                    "display_name": kernel_display_name,
                    "language": language,
                    "name": kernel_name,
                },
                "language_info": {"name": language, "version": language_version},
            }
            self.nbformat = 4
            self.nbformat_minor = 4

        def to_dict(self) -> dict:
            """
            Convert notebook to dictionary format.

            Returns
            -------
            dict
                Notebook in JSON format.
            """
            return {
                "cells": [cell.to_dict() for cell in self.cells],
                "metadata": self.metadata,
                "nbformat": self.nbformat,
                "nbformat_minor": self.nbformat_minor,
            }

        def save(self, path: str | Path, indent: int = 2) -> "Notebook":
            """
            Save notebook to a file.

            Parameters
            ----------
            path : str | Path
                Output file path.
            indent : int, default=2
                JSON indentation level.

            Returns
            -------
            Notebook
                Self for method chaining.

            Examples
            --------
            >>> nb = Notebook()
            >>> nb.save("output.ipynb")
            """
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with path.open("w+") as f:
                json_dump(self.to_dict(), f, indent=indent)

            return self

    banner = (
        "<img src='"
        "data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0id"
        "XRmLTgiPz4KPHN2ZyB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9y"
        "Zy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGx"
        "pbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCAxNzk2IDEwMCIgc3R5bG"
        "U9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgMTc5NiAxMDA7IiB4bWw6c3BhY2U9I"
        "nByZXNlcnZlIj4KPHN0eWxlIHR5cGU9InRleHQvY3NzIj4KCS5zdDB7ZmlsbC1ydWxl"
        "OmV2ZW5vZGQ7Y2xpcC1ydWxlOmV2ZW5vZGQ7ZmlsbDp1cmwoI1NWR0lEXzFfKTt9Cgk"
        "uc3Qxe2ZpbGw6bm9uZTtzdHJva2U6I0ZGRkZGRjtzdHJva2Utd2lkdGg6MjtzdHJva2"
        "UtbWl0ZXJsaW1pdDoxMDt9Cgkuc3Qye2ZpbGw6bm9uZTtzdHJva2U6I0ZGRkZGRjtzd"
        "HJva2Utd2lkdGg6MS41O3N0cm9rZS1taXRlcmxpbWl0OjEwO30KCS5zdDN7ZmlsbDoj"
        "RkZGRkZGO30KCS5zdDR7Zm9udC1mYW1pbHk6J0hlbHZldGljYSBOZXVlJywgQXJpYWw"
        "sIHNhbnMtc2VyaWY7fQoJLnN0NXtmb250LXNpemU6MzJweDt9Cgkuc3Q2e2ZpbGw6Iz"
        "NEM0QzRDt9Cgkuc3Q3e2ZpbGw6IzkzOTU5ODt9Cgkuc3Q4e29wYWNpdHk6MC4yO2Zpb"
        "Gw6dXJsKCNTVkdJRF8yXyk7ZW5hYmxlLWJhY2tncm91bmQ6bmV3O30KCS5zdDl7Zm9u"
        "dC13ZWlnaHQ6NTAwO30KPC9zdHlsZT4KPHJlY3Qgd2lkdGg9IjE3OTYiIGhlaWdodD0"
        "iMTAwIiBmaWxsPSIjMTYxNjE2Ii8+CjxsaW5lYXJHcmFkaWVudCBpZD0iU1ZHSURfMV"
        "8iIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIiB4MT0iNDIuODYiIHkxPSI1M"
        "CIgeDI9Ijc5LjcxIiB5Mj0iNTAiPgoJPHN0b3Agb2Zmc2V0PSIwIiBzdHlsZT0ic3Rv"
        "cC1jb2xvcjojRkY2QjZCIi8+Cgk8c3RvcCBvZmZzZXQ9IjAuMjEiIHN0eWxlPSJzdG9"
        "wLWNvbG9yOiNFRTAwMDAiLz4KCTxzdG9wIG9mZnNldD0iMC43NSIgc3R5bGU9InN0b3"
        "AtY29sb3I6I0NDMDAwMCIvPgoJPHN0b3Agb2Zmc2V0PSIxIiBzdHlsZT0ic3RvcC1jb"
        "2xvcjojQUEwMDAwIi8+CjwvbGluZWFyR3JhZGllbnQ+CjwhLS0gQXV0b1JBRyBJY29u"
        "L0xvZ28gcGxhY2Vob2xkZXIgLSBzaW1wbGlmaWVkIGdlb21ldHJpYyBzaGFwZSAtLT4"
        "KPHBhdGggY2xhc3M9InN0MCIgZD0iTTUyLjQsNDUuOWMwLTIuMywxLjgtNC4xLDQuMS"
        "00LjFzNC4xLDEuOCw0LjEsNC4xUzU4LjgsNTAsNTYuNSw1MGwwLDBjLTIuMiwwLjEtN"
        "C0xLjctNC4xLTMuOQoJQzUyLjQsNDYsNTIuNCw0Niw1Mi40LDQ1Ljl6IE03Ny41LDUy"
        "LjVjLTAuOC0xLjEtMS40LTIuMy0xLjktMy41YzEuMi00LjUsMC43LTguNi0xLjgtMTE"
        "uOWMtMi45LTMuOC04LjItNi0xNC41LTYuMQoJYy00LjUtMC4xLTguOCwxLjctMTIsNC"
        "44Yy0zLDMtNC42LDcuMi00LjUsMTEuNWMtMC4xLDIuOSwwLjksNS44LDIuNyw4LjFjM"
        "C44LDAuOCwxLjMsMS45LDEuNCwzdjQuNWMtMC44LDAuNS0xLjQsMS4zLTEuNCwyLjMK"
        "CWMwLjIsMS41LDEuNSwyLjYsMywyLjRjMS4yLTAuMiwyLjItMS4xLDIuNC0yLjRjMC0"
        "xLTAuNS0xLjktMS40LTIuM3YtNC41YzAtMi0xLTMuMy0xLjktNC42Yy0xLjUtMS45LT"
        "IuMi00LjItMi4xLTYuNQoJYzAtMy41LDEuNC02LjksMy44LTkuNGMyLjctMi43LDYuM"
        "y00LjEsMTAtNC4xYzUuNSwwLDkuOCwxLjksMTIuMSw1YzIsMi44LDIuNSw2LjMsMS40"
        "LDkuNmMtMC40LDEuMiwwLjYsMi43LDIuMyw1LjYKCWMwLjYsMC45LDEuMiwxLjksMS4"
        "2LDIuOWMtMC45LDAuNy0yLDEuMi0zLjEsMS41Yy0wLjUsMC40LTAuNywwLjktMC44LD"
        "EuNVY2NWMwLDAuNC0wLjEsMC44LTAuNCwxLjFjLTAuMywwLjItMC43LDAuMy0xLjEsM"
        "C4zCgljLTEuNi0wLjMtMy40LTAuNy01LjItMS4xdi00LjhjMC44LTAuNSwxLjQtMS40"
        "LDEuNC0yLjNjMC0xLjUtMS4yLTIuNy0yLjctMi43cy0yLjcsMS4yLTIuNywyLjdjMCw"
        "xLDAuNSwxLjksMS40LDIuM3Y0LjEKCWMtMC40LTAuMS0wLjctMC4xLTEuMS0wLjNjLT"
        "QuNS0xLjEtNC41LTIuNi00LjUtMy40di04LjNjMy4yLTAuNyw1LjQtMy41LDUuNS02L"
        "jdjLTAuMS0zLjgtMy4zLTYuNy03LjEtNi42Yy0zLjYsMC4xLTYuNCwzLTYuNiw2LjYK"
        "CWMwLDMuMiwyLjMsNiw1LjUsNi43djguM2MwLDIsMC43LDQuNiw2LjYsNi4xYzMsMC4"
        "4LDYsMS41LDkuMSwxLjljMC4zLDAsMC42LDAuMSwwLjgsMC4xYzEsMCwxLjktMC4zLD"
        "IuNi0xCgljMC45LTAuOCwxLjQtMS45LDEuNC0zLjF2LTQuNWMyLTAuOCw0LjEtMiw0L"
        "jEtMy43Qzc5LjcsNTUuOSw3OSw1NC42LDc3LjUsNTIuNXoiLz4KPGNpcmNsZSBjbGFz"
        "cz0ic3QxIiBjeD0iNTYuNSIgY3k9IjQ1LjkiIHI9IjUuNCIvPgo8Y2lyY2xlIGNsYXN"
        "zPSJzdDIiIGN4PSI0OC4zIiBjeT0iNjUiIHI9IjEuNiIvPgo8Y2lyY2xlIGNsYXNzPS"
        "JzdDIiIGN4PSI2NC44IiBjeT0iNTguMiIgcj0iMS42Ii8+Cjx0ZXh0IHRyYW5zZm9yb"
        "T0ibWF0cml4KDEgMCAwIDEgMTAxLjAyIDU5LjMzKSIgY2xhc3M9InN0MyBzdDQgc3Q1"
        "Ij5BdXRvUkFHPC90ZXh0Pgo8cmVjdCB4PSIyNDIiIHk9IjM0IiBjbGFzcz0ic3Q2IiB"
        "3aWR0aD0iMSIgaGVpZ2h0PSIzMiIvPgo8dGV4dCB0cmFuc2Zvcm09Im1hdHJpeCgxID"
        "AgMCAxIDI1Ni4yOSA1OS42NikiIGNsYXNzPSJzdDcgc3Q0IHN0NSI+UGFydCBvZiBSZ"
        "WQgSGF0IE9wZW5TaGlmdCBBSTwvdGV4dD4KPGxpbmVhckdyYWRpZW50IGlkPSJTVkdJR"
        "F8yXyIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiIHgxPSI3NzMuOCIgeTE9"
        "IjUwIiB4Mj0iMTc5NiIgeTI9IjUwIj4KCTxzdG9wIG9mZnNldD0iMCIgc3R5bGU9InN"
        "0b3AtY29sb3I6IzE2MTYxNiIvPgoJPHN0b3Agb2Zmc2V0PSIwLjUyIiBzdHlsZT0ic3"
        "RvcC1jb2xvcjojRkY2QjZCIi8+Cgk8c3RvcCBvZmZzZXQ9IjAuNjIiIHN0eWxlPSJzd"
        "G9wLWNvbG9yOiNFRTAwMDAiLz4KCTxzdG9wIG9mZnNldD0iMC44OCIgc3R5bGU9InN0"
        "b3AtY29sb3I6I0NDMDAwMCIvPgoJPHN0b3Agb2Zmc2V0PSIxIiBzdHlsZT0ic3RvcC1"
        "jb2xvcjojQUEwMDAwIi8+CjwvbGluZWFyR3JhZGllbnQ+CjxyZWN0IHg9Ijc3My44Ii"
        "BjbGFzcz0ic3Q4IiB3aWR0aD0iMTAyMi4yIiBoZWlnaHQ9IjEwMCIvPgo8dGV4dCB0c"
        "mFuc2Zvcm09Im1hdHJpeCgxIDAgMCAxIDE0MjggNTkuNDYpIiBjbGFzcz0ic3QzIHN0"
        "NCBzdDUgc3Q5Ij5SQUcgUGF0dGVybiBOb3RlYm9vazwvdGV4dD4KPC9zdmc+Cg==' />"
    )

    INDEXING_CELLS_TEMPLATES: dict[
        Literal[
            "BANNER",
            "TABLE_OF_CONTENTS",
            "CHAPTER_1",
            "DEPENDENCIES",
            "MD_1_1",
            "MAIN_IMPORTS",
            "MD_1_2",
            "AWS_ENV",
            "MD_1_3",
            "S3_CLIENT",
            "CHAPTER_2",
            "MD_2_1",
            "LOAD_DATA",
            "MD_2_2",
            "DOCUMENTS_DISCOVERY",
            "MD_2_3",
            "TEXT_EXTRACTION",
            "CHAPTER_3",
            "MD_3_1",
            "LS_CLIENT",
            "MD_3_2",
            "CHUNKER",
            "MD_3_3",
            "VECTOR_STORE",
            "MD_3_4",
            "CHUNKS_UPLOAD",
            "MD_3_5",
            "SAMPLE_SEARCH",
            "SUMMARY",
        ],
        NotebookCell,
    ] = {
        "BANNER": NotebookCell(
            cell_type="markdown",
            source=banner,
        ),
        "TABLE_OF_CONTENTS": NotebookCell(
            cell_type="markdown",
            source=[
                "## Pattern {PATTERN_NAME} Index Building Content\n",
                "\n",
                "This notebook demonstrates how to process documents and build a vector store index for RAG applications. It covers document discovery, text extraction, chunking, and uploading embeddings to a vector database using Llama Stack.\n",
                "\n",
                "### &#x1F4CB; Contents \n",
                "This notebook contains the following sections:\n",
                "\n",
                "- **[Setup](#Setup)**\n",
                "  - [Install packages](#Install-packages)\n",
                "  - [Import required libraries](#Import-required-libraries)\n",
                "  - [Configure S3 credentials](#Configure-S3-credentials)\n",
                "  - [Prepare S3 client](#Prepare-S3-client)\n",
                "- **[Process input documents](#Process-input-documents)**\n",
                "  - [Documents discovery](#Documents-discovery)\n",
                "  - [Text extraction](#Text-extraction)\n",
                "- **[Upload documents content into vector store database](#Upload-documents-content-into-vector-store-database)**\n",
                "  - [Prepare Llama Stack Client](#Prepare-Llama-Stack-Client)\n",
                "  - [Prepare chunker](#Prepare-chunker)\n",
                "  - [Initialize vector store](#Initialize-vector-store)\n",
                "  - [Upload chunks to vector store](#Upload-chunks-to-vector-store)\n",
                "  - [Retrieve chunks for sample question](#Retrieve-chunks-for-sample-question)\n",
                "- **[Summary](#Summary)**",
            ],
        ),
        "CHAPTER_1": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Setup\n",
                "\n",
                "This section sets up the notebook environment by installing required packages, importing libraries, and configuring access to S3 storage.\n",
                "\n",
                "### Install packages\n",
                "\n",
                "Install all required Python packages for document processing and RAG operations:\n",
                "- **boto3**: AWS SDK for Python to interact with S3 storage\n",
                "- **pipelines-components**: Red Hat's pipeline components for data processing\n",
                "- **docling**: Document processing and text extraction library\n",
                "- **ai4rag**: The AutoRAG framework for building RAG applications",
            ],
        ),
        "DEPENDENCIES": NotebookCell(
            cell_type="code",
            source=[
                "!pip install boto3 | tail -n 1\n",
                "!pip install -U --no-cache-dir git+https://github.com/LukaszCmielowski/pipelines-components.git@rhoai_autorag | tail -n 1\n",
                "!pip install docling | tail -n 1\n",
                "!pip install 'ai4rag' | tail -n 1",
            ],
        ),
        "MD_1_1": NotebookCell(
            cell_type="markdown",
            source="### Import required libraries\n\nImport all necessary Python modules and configure logging to suppress verbose output from component loggers.",
        ),
        "MAIN_IMPORTS": NotebookCell(
            cell_type="code",
            source=[
                "import os\n",
                "import json\n",
                "import logging\n",
                "from pathlib import Path\n",
                "from types import SimpleNamespace\n",
                "import getpass\n",
                "\n",
                "import warnings\n",
                'warnings.filterwarnings("ignore")\n',
                "\n",
                "import boto3\n",
                "from langchain_core.documents import Document\n",
                "\n",
                "for logger_name in (\n",
                '        "httpx",\n',
                '        "Document Loader component logger",\n',
                '        "Text Extraction component logger",\n',
                "):\n",
                "    logging.getLogger(logger_name).propagate = False",
            ],
        ),
        "MD_1_2": NotebookCell(
            cell_type="markdown",
            source=[
                "### Configure S3 credentials\n",
                "\n",
                "To load documents from S3-compatible object storage, you need to provide credentials. If you're using OpenShift AI, these can be configured as data connections.\n",
                "\n",
                "&#x1F4CC; **Action**: Provide the credentials for your S3 instance if they are not already set in the notebook environment.\n",
                "\n",
                "&#x1F4A1; **Tip**: In the project, open **Connections** and add an **S3 compatible object storage connection** to a bucket you will use for documents and test data. Open **Workbenches**, edit your workbench, and attach the S3 connection you created so the notebook can read from the bucket. Save and restart the workbench if prompted.",
            ],
        ),
        "AWS_ENV": NotebookCell(
            cell_type="code",
            source=[
                'required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT", "AWS_DEFAULT_REGION", "AWS_S3_BUCKET"]\n',
                "missing = [var for var in required_vars if not os.environ.get(var)]\n",
                "if missing:\n",
                '    raise ValueError(f"Missing required environment variables: {{missing}}")',
            ],
        ),
        "MD_1_3": NotebookCell(
            cell_type="markdown",
            source="### Prepare S3 client\n\nCreates an S3 client session using the provided credentials. This client will be used to discover and download documents from the specified S3 bucket.",
        ),
        "S3_CLIENT": NotebookCell(
            cell_type="code",
            source=[
                "session = boto3.session.Session(\n",
                '    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],\n',
                '    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],\n',
                ")\n",
                "s3_client = session.client(\n",
                "    service_name='s3',\n",
                '    endpoint_url=os.environ["AWS_S3_ENDPOINT"],\n',
                ")",
            ],
        ),
        "CHAPTER_2": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Process input documents\n",
                "\n",
                "This section handles document discovery and text extraction. Documents are first discovered in S3 storage, then their content is extracted and converted to markdown format for further processing.",
            ],
        ),
        "MD_2_1": NotebookCell(
            cell_type="markdown",
            source=[
                "The data processing pipeline prepares documents for the RAG system in multiple steps. Each step runs as a standalone component with outputs stored under `step_outputs/`. \n",
                "\n",
                "| Step | Component | Purpose |\n",
                "|------|-----------|---------|\n",
                "| 1 | **Documents discovery** | List documents in the bucket, prioritize benchmark-referenced docs, apply a size cap, and write a JSON manifest (no content download). |\n",
                "| 2 | **Text extraction** | Download the listed documents from S3 and extract text to Markdown using Docling. |",
            ],
        ),
        "LOAD_DATA": NotebookCell(
            cell_type="code",
            source=[
                "from kfp_components.components.data_processing.autorag.documents_discovery.component import documents_discovery\n",
                "from kfp_components.components.data_processing.autorag.text_extraction.component import text_extraction\n",
                "\n",
                'step_output_dir = Path("./step_outputs")\n',
                "input_data_bucket_name = os.environ['AWS_S3_BUCKET']\n",
                'input_data_key = "{INPUT_DATA_KEY}"\n',
                "step_output_dir.mkdir(parents=True, exist_ok=True)",
            ],
        ),
        "MD_2_2": NotebookCell(
            cell_type="markdown",
            source=[
                "### Documents discovery\n",
                "\n",
                "Lists objects in the S3 input bucket, filters by supported extensions (e.g., `.pdf`, `.docx`, `.pptx`, `.md`, `.html`, `.txt`), and builds a document set. Documents referenced in the benchmark are prioritized, then others are added until a configurable size limit (1 GB by default) is reached. This step does not download document contents but writes a JSON manifest (`documents_descriptor.json`) containing the bucket, prefix, and list of selected object keys and sizes for the next step.",
            ],
        ),
        "DOCUMENTS_DISCOVERY": NotebookCell(
            cell_type="code",
            source=[
                "\n",
                'discovered_documents_out = SimpleNamespace(path=str(step_output_dir / "discovered_documents"))\n',
                "\n",
                "documents_discovery.python_func(\n",
                "    input_data_bucket_name=input_data_bucket_name,\n",
                "    input_data_path=input_data_key,\n",
                "    discovered_documents=discovered_documents_out,\n)\n",
                "\n",
                'descriptor_path = step_output_dir / "discovered_documents" / "documents_descriptor.json"\n',
                "with open(descriptor_path) as f:\n",
                "    descriptor = json.load(f)\n",
                "\n",
                "print(json.dumps(descriptor, indent=4, ensure_ascii=False))",
            ],
        ),
        "MD_2_3": NotebookCell(
            cell_type="markdown",
            source=[
                "### Text extraction\n",
                "\n",
                "Reads the `documents_descriptor.json` produced by the discovery step, downloads each listed document from S3 into a temporary directory, and runs **Docling** to extract text. Output is one Markdown file per document (e.g., `document_0.md`, `document_1.md`) written to the artifact output path. These files form the final text corpus for the RAG system.",
            ],
        ),
        "TEXT_EXTRACTION": NotebookCell(
            cell_type="code",
            source=[
                'descriptor_in = SimpleNamespace(path=str(step_output_dir / "discovered_documents"))\n',
                'extracted_text_out = SimpleNamespace(path=str(step_output_dir / "extracted_text"))\n',
                "\n",
                "text_extraction.python_func(\n",
                "    documents_descriptor=descriptor_in,\n",
                "    extracted_text=extracted_text_out,\n",
                ")",
            ],
        ),
        "CHAPTER_3": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Upload documents content into vector store\n",
                "\n",
                "This section configures the vector store, chunks the extracted documents, and uploads embeddings to the database for semantic search.\n",
                "\n",
                "&#x1F516; **Note**: This notebook requires a Llama Stack server to be available for the AutoRAG experiment. Detailed instructions on how to setup Llama Stack server for AutoRAG can be found here: https://github.com/LukaszCmielowski/prototypes/blob/main/llamastack/SETUP.md",
            ],
        ),
        "MD_3_1": NotebookCell(
            cell_type="markdown",
            source=[
                "### Prepare Llama Stack Client\n",
                "\n",
                "The Llama Stack client provides the interface to the embedding models and vector database. This section initializes the client using API credentials from environment variables or prompts.\n",
                "\n",
                "**Prerequisites:**\n",
                "- `LLAMA_STACK_CLIENT_API_KEY`: Your authentication key for the Llama Stack API\n",
                "- `LLAMA_STACK_CLIENT_BASE_URL`: The base URL of your Llama Stack instance\n",
                "\n",
                "&#x1F4A1; **Tip**: In OpenShift AI Workbench, you can add these as environment variables or data connections to avoid entering them manually each time.",
            ],
        ),
        "LS_CLIENT": NotebookCell(
            cell_type="code",
            source=[
                "from llama_stack_client import LlamaStackClient\n",
                "\n",
                'if not os.getenv("LLAMA_STACK_CLIENT_API_KEY") or not os.getenv("LLAMA_STACK_CLIENT_BASE_URL"):\n',
                '    os.environ["LLAMA_STACK_CLIENT_API_KEY"] = getpass.getpass("Please enter \'LLAMA_STACK_CLIENT_API_KEY\': ")\n',
                '    os.environ["LLAMA_STACK_CLIENT_BASE_URL"] = getpass.getpass("Please enter \'LLAMA_STACK_CLIENT_BASE_URL\': ")\n',
                "\n",
                "client = LlamaStackClient(\n",
                '    base_url=os.getenv("LLAMA_STACK_CLIENT_BASE_URL"),\n',
                '    api_key=os.getenv("LLAMA_STACK_CLIENT_API_KEY"),\n',
                ")",
            ],
        ),
        "MD_3_2": NotebookCell(
            cell_type="markdown",
            source=[
                "### Prepare chunker\n",
                "\n",
                "The chunker splits extracted documents into smaller chunks for more effective retrieval. Configuration includes:\n",
                "- **Chunking Method**: The algorithm used to split text (e.g., recursive character splitting)\n",
                "- **Chunk Size**: Maximum number of characters per chunk\n",
                "- **Chunk Overlap**: Number of overlapping characters between consecutive chunks to preserve context\n",
                "\n",
                "Proper chunking ensures that retrieved context is both relevant and fits within the model's context window.",
            ],
        ),
        "CHUNKER": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.chunking import LangChainChunker\n",
                "\n",
                'chunking_method = "{CHUNKING_METHOD}"\n',
                "chunk_size = {CHUNK_SIZE}\n",
                "chunk_overlap = {CHUNK_OVERLAP}\n",
                "\n",
                "chunker = LangChainChunker(method=chunking_method, chunk_size=chunk_size, chunk_overlap=chunk_overlap)",
            ],
        ),
        "MD_3_3": NotebookCell(
            cell_type="markdown",
            source=[
                "### Initialize vector store\n",
                "\n",
                "The vector store manages document embeddings and enables semantic search. This section configures:\n",
                "- **Embedding Model**: Converts text chunks into vector representations\n",
                "- **Vector Database Provider**: The backend storage system (e.g., Milvus)\n",
                "- **Distance Metric**: How similarity is calculated (cosine, euclidean, etc.)\n",
                "- **Collection Name**: A named collection where embeddings are stored\n",
                "\n",
                "The vector store is initialized and ready to receive document chunks.",
            ],
        ),
        "VECTOR_STORE": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel, LSEmbeddingParams\n",
                "from ai4rag.rag.vector_store.llama_stack import LSVectorStore\n",
                "\n",
                'model_id = "{MODEL_ID}"\n',
                "params = LSEmbeddingParams(**{EMBEDDING_PARAMS})\n",
                "\n",
                "embedding_model = LSEmbeddingModel(client=client, model_id=model_id, params=params)\n",
                "\n",
                'provider_id = "{PROVIDER_ID}"\n',
                'distance_metric = "{DISTANCE_METRIC}"\n',
                'collection_name = "{COLLECTION_NAME}"\n',
                "\n",
                "ls_vectorstore = LSVectorStore(\n",
                "    embedding_model=embedding_model,\n",
                "    client=client,\n",
                "    provider_id=provider_id,\n",
                "    distance_metric=distance_metric,\n",
                "    reuse_collection_name=collection_name\n",
                ")",
            ],
        ),
        "MD_3_4": NotebookCell(
            cell_type="markdown",
            source=[
                "### Upload chunks to vector store\n",
                "\n",
                "This section processes each extracted markdown file by:\n",
                "- Loading the document content with metadata\n",
                "- Splitting it into chunks using the configured chunker\n",
                "- Generating embeddings and uploading them to the vector store\n",
                "\n",
                "Once complete, all document chunks are indexed and ready for semantic search queries.",
            ],
        ),
        "CHUNKS_UPLOAD": NotebookCell(
            cell_type="code",
            source=[
                'paths = list(Path("step_outputs/extracted_text").glob("*.md"))\n',
                "\n",
                "for p in sorted(paths):\n",
                "    document = Document(\n",
                '            page_content=p.read_text(encoding="utf-8", errors="replace"),\n',
                '            metadata={{"document_id": p.stem}},\n',
                "        )\n",
                "\n",
                "    chunked_documents = chunker.split_documents([document])\n",
                "    ls_vectorstore.add_documents(chunked_documents)",
            ],
        ),
        "MD_3_5": NotebookCell(
            cell_type="markdown",
            source="### Retrieve chunks for sample question\n\nThis section demonstrates how to perform a semantic search query against the populated vector store. You can test retrieval by searching for relevant chunks based on a sample question.",
        ),
        "SAMPLE_SEARCH": NotebookCell(
            cell_type="code",
            source=[
                "from pprint import pprint\n",
                "\n",
                "sample_question = input()\n",
                "\n",
                "results = ls_vectorstore.search(query=sample_question, k=5)\n",
                "for result in results:\n",
                "    if isinstance(result, tuple):\n",
                "        pprint(result[0].model_dump(mode='python'), indent=4)\n",
                "        continue\n",
                "    pprint(result.model_dump(mode='python'), indent=4)",
            ],
        ),
        "SUMMARY": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Summary\n",
                "\n",
                "This notebook successfully processed documents from S3 storage, extracted their text content using Docling, chunked the text into manageable pieces, and uploaded the embeddings to a vector store. The indexed documents are now ready for semantic search and retrieval in RAG applications.",
            ],
        ),
    }

    GENERATION_CELLS_TEMPLATES = {
        "BANNER": NotebookCell(cell_type="markdown", source=banner),
        "TABLE_OF_CONTENTS": NotebookCell(
            cell_type="markdown",
            source=[
                "## Pattern {PATTERN_NAME} Retrieve & Generation Content\n",
                "\n",
                "This notebook demonstrates how to implement and test a Retrieval-Augmented Generation (RAG) pattern using Llama Stack. It guides you through setting up the necessary components, loading test data from an S3 bucket, and querying the RAG system to generate responses based on retrieved context.\n",
                "\n",
                "&#x26A0;&#xFE0F; **Important**: Before running this notebook, you must first run the corresponding **indexing.ipynb** notebook to populate the vector store with document embeddings. The indexing process prepares the knowledge base that this notebook queries.\n",
                "\n",
                "### &#x1F4CB; Contents \n",
                "This notebook contains the following sections:\n",
                "\n",
                "- **[Setup](#Setup)**\n",
                "- **[Prepare LlamaStackClient](#Prepare-LlamaStackClient)**\n",
                "- **[Initialize RAG Components](#Initialize-RAG-Components)**\n",
                "   - [Initialize LlamaStack Foundation Model](#Initialize-LlamaStack-Foundation-Model)\n",
                "   - [Initialize Vector Store Client](#Initialize-Vector-Store-Client)\n",
                "   - [Initialize Retriever](#Initialize-Retriever)\n",
                "   - [Initialize RAG Pattern](#Initialize-RAG-Pattern)\n",
                "   - [Query RAG Pattern](#Query-RAG-Pattern)\n",
                "- **[Next steps](#Next-steps)**\n",
                "   - [Load Test Data](#Load-Test-Data)\n",
                "   - [Configure S3 Credentials](#Configure-S3-Credentials)\n",
                "   - [Initialize S3 Client](#Initialize-S3-Client)\n",
                "   - [Load Benchmark Data](#Load-Benchmark-Data)\n",
                "   - [Build Evaluation Data](#Build-Evaluation-Data)\n",
                "   - [Evaluate Response](#Evaluate-Response)\n",
                "- **[Summary](#Summary)**",
            ],
        ),
        "CHAPTER_1": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Setup\n",
                "\n",
                "This section installs all the required Python packages for running the RAG experiment:\n",
                "- **boto3**: AWS SDK for Python to interact with S3 storage\n",
                "- **pipelines-components**: Red Hat's pipeline components for data processing\n",
                "- **ai4rag**: The main RAG framework for AutoRAG experiments",
            ],
        ),
        "DEPENDENCIES": NotebookCell(
            cell_type="code",
            source=[
                "!pip install boto3 | tail -n 1\n",
                "!pip install -U --no-cache-dir git+https://github.com/LukaszCmielowski/pipelines-components.git@rhoai_autorag | tail -n 1\n",
                "!pip install 'ai4rag' | tail -n 1",
            ],
        ),
        "CHAPTER_2": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Prepare LlamaStackClient\n",
                "\n",
                "The Llama Stack client is the core interface for interacting with the Llama Stack API. This section initializes the client by:\n",
                "- Retrieving API credentials from environment variables or prompting for them\n",
                "- Establishing a connection to the Llama Stack endpoint\n",
                "\n",
                "**Prerequisites:**\n",
                "- `LLAMA_STACK_CLIENT_API_KEY`: Your authentication key for the Llama Stack API\n",
                "- `LLAMA_STACK_CLIENT_BASE_URL`: The base URL of your Llama Stack instance\n",
                "\n",
                "&#x1F4A1; **Tip**: In OpenShift AI Workbench, you can add these as environment variables or data connections to avoid entering them manually each time.",
            ],
        ),
        "LS_CLIENT": NotebookCell(
            cell_type="code",
            source=[
                "import os\n",
                "import getpass\n",
                "import warnings\n",
                "import logging\n",
                "\n",
                "from llama_stack_client import LlamaStackClient\n",
                "\n",
                'warnings.filterwarnings("ignore")\n',
                "logging.getLogger('httpx').propagate = False\n",
                "\n",
                'if not os.getenv("LLAMA_STACK_CLIENT_API_KEY") or not os.getenv("LLAMA_STACK_CLIENT_BASE_URL"):\n',
                '    os.environ["LLAMA_STACK_CLIENT_API_KEY"] = getpass.getpass("Please enter \'LLAMA_STACK_CLIENT_API_KEY\': ")\n',
                '    os.environ["LLAMA_STACK_CLIENT_BASE_URL"] = getpass.getpass("Please enter \'LLAMA_STACK_CLIENT_BASE_URL\': ")\n',
                "\n",
                "client = LlamaStackClient(\n",
                '    base_url=os.getenv("LLAMA_STACK_CLIENT_BASE_URL"),\n',
                '    api_key=os.getenv("LLAMA_STACK_CLIENT_API_KEY"),\n',
                ")",
            ],
        ),
        "CHAPTER_3": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Initialize RAG Components\n",
                "\n",
                "This section sets up all the components needed for the RAG pattern: foundation model, vector store, retriever, and the RAG pattern itself.",
            ],
        ),
        "MD_3_1": NotebookCell(
            cell_type="markdown",
            source=[
                "### Initialize LlamaStack Foundation Model\n",
                "\n",
                "The foundation model is the core language model that generates responses. This section configures:\n",
                "- **Model ID**: The specific Llama model to use for generation\n",
                "- **System Message**: Instructions that define the model's behavior and role\n",
                "- **User Message Template**: The format for user queries\n",
                "- **Context Template**: How retrieved context is incorporated into prompts\n",
                "\n",
                "These templates control how the RAG system structures prompts to the language model.",
            ],
        ),
        "FOUNDATION_MODEL": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel\n",
                "\n",
                'chat_model_id = """{FM_MODEL_ID}"""\n',
                'system_message_text = """{SYSTEM_MESSAGE}"""\n',
                'user_message_text = """{USER_MESSAGE}"""\n',
                'context_template_text = """{CONTEXT_TEXT}"""\n',
                "\n",
                "lsfoundationmodel = LSFoundationModel(\n",
                "    client=client,\n",
                "    model_id=chat_model_id,\n",
                "    system_message_text=system_message_text,\n",
                "    user_message_text=user_message_text,\n",
                "    context_template_text=context_template_text,\n",
                ")",
            ],
        ),
        "MD_3_2": NotebookCell(
            cell_type="markdown",
            source=[
                "### Initialize Vector Store Client\n",
                "\n",
                "The vector store is responsible for storing and retrieving document embeddings. This section sets up:\n",
                "- **Embedding Model**: Converts text into vector representations for semantic search\n",
                "- **Vector Database**: Stores embeddings with configurable distance metrics (cosine, euclidean, etc.)\n",
                "- **Collection**: A named collection where document vectors are stored and can be reused\n",
                "\n",
                "The vector store enables semantic similarity search to find relevant context for user queries.",
            ],
        ),
        "VECTOR_STORE": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel, LSEmbeddingParams\n",
                "from ai4rag.rag.vector_store.llama_stack import LSVectorStore\n",
                "\n",
                'embedding_model_id = "{MODEL_ID}"\n',
                "params = LSEmbeddingParams(**{EMBEDDING_PARAMS})\n",
                "embedding_model = LSEmbeddingModel(client=client, model_id=embedding_model_id, params=params)\n",
                'provider_id = "{PROVIDER_ID}"\n',
                'distance_metric = "{DISTANCE_METRIC}"\n',
                'collection_name = "{COLLECTION_NAME}"\n',
                "\n",
                "ls_vectorstore = LSVectorStore(\n",
                "    embedding_model=embedding_model,\n",
                "    client=client,\n",
                "    provider_id=provider_id,\n",
                "    distance_metric=distance_metric,\n",
                "    reuse_collection_name=collection_name\n",
                ")",
            ],
        ),
        "MD_3_3": NotebookCell(
            cell_type="markdown",
            source=[
                "### Initialize Retriever\n",
                "\n",
                "The retriever finds the most relevant document chunks for a given query. Configuration includes:\n",
                "- **Retrieval Method**: The algorithm used to find relevant documents (e.g., similarity search, hybrid search)\n",
                "- **Number of Chunks**: How many document chunks to retrieve and include in the context\n",
                "\n",
                "The retriever acts as the bridge between user questions and the knowledge base.",
            ],
        ),
        "RETRIEVER": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.retrieval.retriever import Retriever\n",
                "\n",
                'method = "{RETRIEVAL_METHOD}"\n',
                "number_of_chunks = {NUMBER_OF_CHUNKS}\n",
                "\n",
                "retriever = Retriever(vector_store=ls_vectorstore, method=method, number_of_chunks=number_of_chunks)",
            ],
        ),
        "MD_3_4": NotebookCell(
            cell_type="markdown",
            source=[
                "### Initialize RAG Pattern\n",
                "\n",
                "This section brings together all components into a complete RAG pattern:\n",
                "- Combines the foundation model with the retriever\n",
                "- Creates a unified interface for question-answering\n",
                "- Coordinates the retrieve-then-generate workflow\n",
                "\n",
                "The RAG pattern orchestrates: query, retrieve context, generate response.",
            ],
        ),
        "RAG_PATTERN": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.template.llama_stack_rag_template import LlamaStackRAG\n",
                "\n",
                "rag_pattern = LlamaStackRAG(foundation_model=lsfoundationmodel, retriever=retriever)",
            ],
        ),
        "MD_3_4": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "### Query RAG Pattern\n",
                "\n",
                "This section executes the RAG workflow by submitting test questions to the system and generating responses based on retrieved context.",
            ],
        ),
        "TEST_RESPONSE": NotebookCell(
            cell_type="code",
            source=[
                "from pprint import pprint\n",
                "\n",
                "question = input()\n",
                "response = rag_pattern.generate(question=question)\n",
                "pprint(response, indent=4, width=50)",
            ],
        ),
        "CHAPTER_4": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Next steps\n",
                "\n",
                "The following sections provide optional next steps for loading test data, running queries, and evaluating the RAG pattern's performance. These steps are useful for systematic testing and benchmarking, but can be skipped if you prefer to interact with the RAG system directly using the pattern configured above.",
            ],
        ),
        "MD_4_1": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "### Load Test Data\n",
                "\n",
                "This section prepares the test environment and loads benchmark questions from S3 storage. The test data is used to evaluate the RAG system's performance.",
            ],
        ),
        "LOAD_DATA_IMPORTS": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                "import os\n",
                "import json\n",
                "from pathlib import Path\n",
                "from types import SimpleNamespace\n",
                "\n",
                "import boto3\n",
                "\n",
                "logging.getLogger('Test Data Loader component logger').propagate = False\n",
                "```",
            ],
        ),
        "MD_4_2": NotebookCell(
            cell_type="markdown",
            source=[
                "### Configure S3 Credentials\n",
                "\n",
                "To load test data from S3-compatible object storage, you need to provide credentials. If you're using OpenShift AI, these can be configured as data connections.\n",
                "\n",
                "&#x1F4CC; **Action**: Provide the credentials for your S3 instance if they are not already set in the notebook environment.\n",
                "\n",
                "&#x1F4A1; **Tip**: In the project, open **Connections** and add an **S3 compatible object storage connection** to a bucket you will use for documents and test data. Open **Workbenches**, edit your workbench, and attach the S3 connection you created so the notebook can read from the bucket. Save and restart the workbench if prompted.",
            ],
        ),
        "AWS_ENV": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                'required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT", "AWS_DEFAULT_REGION", "AWS_S3_BUCKET"]\n',
                "missing = [var for var in required_vars if not os.environ.get(var)]\n",
                "if missing:\n",
                '    raise ValueError(f"Missing required environment variables: {{missing}}")\n',
                "```",
            ],
        ),
        "MD_4_3": NotebookCell(
            cell_type="markdown",
            source="### Initialize S3 Client\n\nCreates an S3 client session using the provided credentials. This client is used to download test data from the specified S3 bucket.",
        ),
        "S3_CLIENT": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                "session = boto3.session.Session(\n",
                '    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],\n',
                '    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],\n',
                ")\n",
                "s3_client = session.client(\n",
                "    service_name='s3',\n",
                '    endpoint_url=os.environ["AWS_S3_ENDPOINT"],\n',
                ")\n",
                "```",
            ],
        ),
        "MD_4_4": NotebookCell(
            cell_type="markdown",
            source=[
                "### Load Benchmark Data\n",
                "\n",
                "Downloads and loads the benchmark test data from S3. The benchmark file should be a JSON file containing:\n",
                "- **question**: The test question to ask the RAG system\n",
                "- **correct_answers**: The expected answers for evaluation\n",
                "- **correct_answer_document_ids**: IDs of documents that contain the correct information\n",
                "\n",
                "This data is used to measure the RAG system's accuracy and retrieval quality.",
            ],
        ),
        "TEST_DATA_LOADER": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                "from kfp_components.components.data_processing.autorag.test_data_loader.component import test_data_loader\n",
                "\n",
                "\n",
                'step_output_dir = Path("./step_outputs")\n',
                "step_output_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "test_data_bucket_name = os.environ['AWS_S3_BUCKET']\n",
                'test_data_key = "{TEST_DATA_KEY}"\n',
                'test_data_out = SimpleNamespace(path=str(step_output_dir / "test_data.json"))\n',
                "\n",
                "test_data_loader.python_func(\n",
                "    test_data_bucket_name=test_data_bucket_name,\n",
                "    test_data_path=test_data_key,\n",
                "    test_data=test_data_out,\n",
                ")\n",
                "\n",
                "output_path = Path(test_data_out.path)\n",
                'with output_path.open("r", encoding="utf-8") as f:\n',
                "    test_data = json.load(f)\n",
                "\n",
                "print(json.dumps(test_data, indent=4, ensure_ascii=False))\n",
                "```",
            ],
        ),
        "EXECUTE_QUERIES": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                "inference_responses = []\n",
                "\n",
                "for test_data_item in test_data:\n",
                '    response = rag_pattern.generate(question=test_data_item["question"])\n',
                "    inference_responses.append(response)\n",
                "```",
            ],
        ),
        "MD_4_5": NotebookCell(
            cell_type="markdown",
            source=[
                "### Build Evaluation Data\n",
                "\n",
                "This section transforms the RAG system's inference responses into a structured format for evaluation. It combines:\n",
                "- **Benchmark Data**: The original test questions and expected answers\n",
                "- **Inference Responses**: The actual responses generated by the RAG system\n",
                "\n",
                "The resulting evaluation data structure allows for systematic comparison between expected and actual outputs, enabling metric calculation for assessing the RAG system's performance.",
            ],
        ),
        "BUILD_EVAL_DATA": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                "from pandas import DataFrame\n",
                "\n",
                "from ai4rag.core.experiment.utils import build_evaluation_data\n",
                "from ai4rag.core.experiment.benchmark_data import BenchmarkData\n",
                "\n",
                "\n",
                "benchmark_data = BenchmarkData(\n",
                "    DataFrame(\n",
                "        data=test_data\n",
                "    )\n",
                ")\n",
                "\n",
                "evaluation_data = build_evaluation_data(\n",
                "    benchmark_data=benchmark_data, \n",
                "    inference_response=inference_responses\n",
                ")\n",
                "```",
            ],
        ),
        "MD_4_6": NotebookCell(
            cell_type="markdown",
            source=[
                "### Evaluate Response\n",
                "\n",
                "This section evaluates the quality of the RAG system's responses by comparing them against the expected answers from the benchmark data. Evaluation metrics may include accuracy, relevance, and retrieval precision.",
            ],
        ),
        "EVALUATE_RESPONSE": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                "from ai4rag.evaluator.unitxt_evaluator import UnitxtEvaluator\n",
                "from ai4rag.evaluator.base_evaluator import MetricType\n",
                "\n",
                "evaluator = UnitxtEvaluator()\n",
                "evaluator.evaluate_metrics(evaluation_data=evaluation_data, metrics=(MetricType.ANSWER_CORRECTNESS, MetricType.FAITHFULNESS, MetricType.CONTEXT_CORRECTNESS))\n",
                "```",
            ],
        ),
        "SUMMARY": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Summary\n",
                "\n",
                "This notebook successfully demonstrates a complete RAG pattern implementation using Llama Stack, from initializing the foundation model and vector store to querying the system with test data. The evaluation framework allows you to measure the quality of generated responses against benchmark answers using multiple metrics including answer correctness, faithfulness, and context correctness.",
            ],
        ),
    }

    def create_placeholder_mapping(
        output_data: dict[str, Any],
        test_data_key: str = "",
        input_data_key: str = "",
    ) -> dict[str, Any]:
        """
        Create a mapping from placeholder names to their values from output.json.

        This function extracts values from the output.json structure and creates
        a flat dictionary suitable for use with NotebookCell.format_source().

        Expected output.json structure:
        {
            "config": {
                "pattern_name": "...",
                "autorag_version": "...",
                "llama_stack": {
                    "foundation_model": {...},
                    "embedding_model": {...},
                    "vector_store": {...},
                    "retriever": {...},
                    "chunker": {...}
                },
                "data": {...}
            }
        }

        Args:
            output_data: The parsed output.json data

        Returns:
            Dictionary mapping placeholder names to their values
        """
        mapping = {}

        mapping["PATTERN_NAME"] = output_data.get("name", "")
        settings = output_data.get("settings", {})
        fm = settings.get("generation", {})
        mapping["FM_MODEL_ID"] = fm.get("model_id", "")
        mapping["SYSTEM_MESSAGE"] = fm.get("system_message_text", "")
        mapping["USER_MESSAGE"] = fm.get("user_message_text", "")
        mapping["CONTEXT_TEXT"] = fm.get("context_template_text", "")

        em = settings.get("embedding", {})
        mapping["MODEL_ID"] = em.get("model_id", "")
        mapping["EMBEDDING_PARAMS"] = em.get("embedding_params", {"embedding_dimension": 768})
        mapping["DISTANCE_METRIC"] = em.get("distance_metric", "")

        vs = settings.get("vector_store", {})
        mapping["PROVIDER_ID"] = vs.get("datasource_type", "")
        mapping["COLLECTION_NAME"] = vs.get("collection_name", "")

        ret = settings.get("retrieval", {})
        mapping["RETRIEVAL_METHOD"] = ret.get("method", "")
        mapping["NUMBER_OF_CHUNKS"] = ret.get("number_of_chunks", 5)

        ch = settings.get("chunking", {})
        mapping["CHUNKING_METHOD"] = ch.get("method", "")
        mapping["CHUNK_SIZE"] = ch.get("chunk_size", 512)
        mapping["CHUNK_OVERLAP"] = ch.get("chunk_overlap", 50)

        mapping["TEST_DATA_KEY"] = test_data_key
        mapping["INPUT_DATA_KEY"] = input_data_key

        return mapping

    def generate_notebook_from_templates(
        templates_dict: dict[str, NotebookCell],
        output_data: dict[str, Any],
        output_notebook_path: Path,
        test_data_key: str = "",
        input_data_key: str = "",
    ) -> None:
        """
        Generate a filled notebook from templates and output.json.

        Args:
            templates_dict: Dictionary of NotebookCell templates (e.g., INDEXING_CELLS_TEMPLATES)
            output_json_path: Path to the output.json file
            output_notebook_path: Path where to save the generated notebook
            test_data_key: A path to test data file within a bucket used as an input to AI4RAG experiment.
            input_data_key: A path to documents dir within a bucket used as an input to AI4RAG experiment.

        Returns:
            The generated Notebook object
        """

        placeholder_mapping = create_placeholder_mapping(
            output_data, test_data_key=test_data_key, input_data_key=input_data_key
        )
        templates_dict_copy = deepcopy(templates_dict)

        filled_cells = []
        for cell in templates_dict_copy.values():
            filled_cell = cell.format_source(placeholder_mapping)
            filled_cells.append(filled_cell)

        notebook = Notebook(cells=filled_cells)

        notebook.save(Path(output_notebook_path))

    if embedding_model_url and chat_model_url:
        # Specification of OpenAI API compatibility
        embedding_model_url += "/v1"
        chat_model_url += "/v1"

    class TmpEventHandler(BaseEventHandler):
        """Exists temporarily only for the purpose of satisying type hinting checks"""

        def on_status_change(self, level: LogLevel, message: str, step: str | None = None) -> None:
            pass

        def on_pattern_creation(self, payload: dict, evaluation_results: list, **kwargs) -> None:
            pass

    def load_as_langchain_doc(path: str | Path) -> list[Document]:
        """Load a text file or folder into a list of langchain Document objects.

        Args:
            path: A local path to either a text file or a folder of text files.

        Returns:
            A list of langchain `Document` objects.
        """
        if isinstance(path, str):
            path = Path(path)

        documents = []
        if path.is_dir():
            for doc_path in path.iterdir():
                with doc_path.open("r", encoding="utf-8") as doc:
                    documents.append(
                        Document(
                            page_content=doc.read(),
                            metadata={"document_id": doc_path.stem},
                        )
                    )

        elif path.is_file():
            with path.open("r", encoding="utf-8") as doc:
                documents.append(Document(page_content=doc.read(), metadata={"document_id": path.stem}))

        return documents

    llama_stack_client_base_url = os.environ.get("LLAMA_STACK_CLIENT_BASE_URL", None)
    llama_stack_client_api_key = os.environ.get("LLAMA_STACK_CLIENT_API_KEY", None)

    in_memory_vector_store_scenario = False
    Client = namedtuple(
        "Client",
        ["llama_stack", "generation_model", "embedding_model"],
        defaults=[None, None, None],
    )

    if llama_stack_client_base_url and llama_stack_client_api_key:
        client = Client(llama_stack=LlamaStackClient())
    else:
        if not all(
            (
                chat_model_url,
                chat_model_token,
                embedding_model_url,
                embedding_model_token,
            )
        ):
            raise ValueError(
                "All of (`chat_model_url`, `chat_model_token`, `embedding_model_url`, `embedding_model_token`) "
                "have to be defined when running AutoRAG experiment using an in-memory vector store."
            )
        client = Client(
            generation_model=OpenAI(api_key=chat_model_token, base_url=chat_model_url),
            embedding_model=OpenAI(api_key=embedding_model_token, base_url=embedding_model_url),
        )
        in_memory_vector_store_scenario = True

    def construct_model_instance(loader, node: yml.MappingNode) -> BaseEmbeddingModel | BaseFoundationModel:
        """Instructs yml.Loader on how to construct "!Model" tag."""
        mapping = loader.construct_mapping(node, deep=True)

        match mapping:
            case {"type_": "embedding", **id_to_params}:
                model_id, params = id_to_params.popitem()
                if in_memory_vector_store_scenario:
                    return OpenAIEmbeddingModel(client=client.embedding_model, model_id=model_id, params=params)
                else:
                    return LSEmbeddingModel(client=client.llama_stack, model_id=model_id, params=params)

            case {"type_": "generation", **id_to_params}:
                model_id, params = id_to_params.popitem()
                if in_memory_vector_store_scenario:
                    return OpenAIFoundationModel(client=client.generation_model, model_id=model_id, params=params)
                else:
                    return LSFoundationModel(client=client.llama_stack, model_id=model_id, params=params)
            case _:
                raise ValueError(f"Cannot load the yml-serialized !Model tag: {mapping}")

    yml.add_constructor("!Model", construct_model_instance, Loader=yml.SafeLoader)

    optimization_settings = optimization_settings if optimization_settings else {}
    if not (optimization_metric := optimization_settings.get("metric", None)):
        optimization_metric = METRIC
    if optimization_metric not in SUPPORTED_OPTIMIZATION_METRICS:
        raise ValueError(
            "optimization_metric must be one of %s; got %r"
            % (sorted(SUPPORTED_OPTIMIZATION_METRICS), optimization_metric)
        )

    documents = load_as_langchain_doc(extracted_text)

    # reload the search space
    with open(search_space_prep_report, "r") as f:
        search_space = yml.safe_load(f)

    search_space = AI4RAGSearchSpace(
        params=[Parameter(param, "C", values=values) for param, values in search_space.items()]
    )

    event_handler = TmpEventHandler()
    max_rag_patterns = optimization_settings.get("max_number_of_rag_patterns", MAX_NUMBER_OF_RAG_PATTERNS)
    optimizer_settings = GAMOptSettings(max_evals=int(max_rag_patterns))

    benchmark_data = pd.read_json(Path(test_data))

    if not llama_stack_vector_database_id and in_memory_vector_store_scenario:
        llama_stack_vector_database_id = "chroma"
    elif not llama_stack_vector_database_id:
        llama_stack_vector_database_id = "ls_milvus"

    rag_exp = AI4RAGExperiment(
        client=None if in_memory_vector_store_scenario else client.llama_stack,
        event_handler=event_handler,
        optimizer_settings=optimizer_settings,
        search_space=search_space,
        benchmark_data=benchmark_data,
        vector_store_type=llama_stack_vector_database_id,
        documents=documents,
        optimization_metric=optimization_metric,
        # TODO some necessary kwargs (if any at all)
    )

    # retrieve documents && run optimisation loop
    rag_exp.search()

    def _evaluation_result_fallback(eval_data_list, evaluation_result):
        """Build evaluation_results.json-style list when question_scores missing or incomplete."""
        out = []
        for ev in eval_data_list:
            answer_contexts = []
            if getattr(ev, "contexts", None) and getattr(ev, "context_ids", None):
                answer_contexts = [{"text": t, "document_id": doc_id} for t, doc_id in zip(ev.contexts, ev.context_ids)]
            scores = {}
            q_scores = (evaluation_result.scores or {}).get("question_scores") or {}
            for key in q_scores:
                if isinstance(q_scores[key], dict) and getattr(ev, "question_id", None) in q_scores[key]:
                    scores[key] = q_scores[key][ev.question_id]
            out.append(
                {
                    "question": getattr(ev, "question", ""),
                    "correct_answers": getattr(ev, "ground_truths", None),
                    "answer": getattr(ev, "answer", ""),
                    "answer_contexts": answer_contexts,
                    "scores": scores,
                }
            )
        return out

    rag_patterns_dir = Path(rag_patterns.path)
    evaluation_data_list = getattr(rag_exp.results, "evaluation_data", [])

    def _build_pattern_json(evaluation_result, iteration: int, max_combinations: int) -> dict:
        """Build pattern.json with flat schema (name, iteration, settings, scores, final_score)."""
        idx = evaluation_result.indexing_params or {}
        rp = evaluation_result.rag_params or {}
        chunking = idx.get("chunking") or {}
        # ai4rag puts embedding in indexing_params.embedding, not rag_params
        embedding_from_idx = idx.get("embedding") or idx.get("embeddings") or {}
        embeddings = rp.get("embeddings") or rp.get("embedding") or embedding_from_idx
        retrieval = rp.get("retrieval") or {}
        generation = rp.get("generation") or {}
        # embedding model_id: from indexing_params.embedding (ai4rag), or rag_params, or flat embedding_model
        embedding_model_id = None
        if isinstance(embedding_from_idx, dict) and embedding_from_idx.get("model_id"):
            embedding_model_id = embedding_from_idx.get("model_id")
        if not embedding_model_id and isinstance(embeddings, dict):
            embedding_model_id = embeddings.get("model_id")
        if not embedding_model_id and isinstance(rp.get("embedding_model"), str):
            embedding_model_id = rp.get("embedding_model")
        if not embedding_model_id and hasattr(rp.get("embedding_model"), "model_id"):
            embedding_model_id = getattr(rp.get("embedding_model"), "model_id", None)
        # generation model_id: from rag_params.generation (ai4rag) or flat foundation_model
        generation_model_id = generation.get("model_id") if isinstance(generation, dict) else None
        if not generation_model_id and isinstance(rp.get("foundation_model"), str):
            generation_model_id = rp.get("foundation_model")
        if not generation_model_id and hasattr(rp.get("foundation_model"), "model_id"):
            generation_model_id = getattr(rp.get("foundation_model"), "model_id", None)
        return {
            "name": getattr(evaluation_result, "pattern_name", ""),
            "iteration": iteration,
            "max_combinations": max_combinations,
            "duration_seconds": getattr(evaluation_result, "execution_time", 0) or 0,
            "settings": {
                "vector_store": {
                    "datasource_type": idx.get("vector_store", {}).get("datasource_type")
                    or rp.get("vector_store", {}).get("datasource_type")
                    or "ls_milvus",
                    "collection_name": getattr(evaluation_result, "collection", "") or "",
                },
                "chunking": {
                    "method": chunking.get("method", "recursive"),
                    "chunk_size": chunking.get("chunk_size", 2048),
                    "chunk_overlap": chunking.get("chunk_overlap", 256),
                },
                "embedding": {
                    "model_id": embedding_model_id or "",
                    "distance_metric": (
                        embeddings.get("distance_metric", "cosine") if isinstance(embeddings, dict) else "cosine"
                    ),
                    "embedding_params": embeddings.get("embedding_params", {"embedding_dimension": 768}),
                },
                "retrieval": {
                    "method": retrieval.get("method", "simple"),
                    "number_of_chunks": retrieval.get("number_of_chunks", 5),
                },
                "generation": {
                    "model_id": generation_model_id or "",
                    "context_template_text": generation.get("context_template_text", "{document}"),
                    "user_message_text": generation.get(
                        "user_message_text",
                        (
                            "\n\nContext:\n{reference_documents}:\n\nQuestion: {question}. \nAgain, please answer "
                            "the question based on the context provided only. If the context is not related to "
                            "the question, just say you cannot answer. Respond exclusively in the language of "
                            "the question."
                        ),
                    ),
                    "system_message_text": generation.get(
                        "system_message_text",
                        (
                            "Please answer the question I provide in the Question section below, based solely "
                            "on the information I provide in the Context section. If unanswerable, say so."
                        ),
                    ),
                },
            },
        }

    evaluations_list = list(rag_exp.results.evaluations)
    max_combinations = getattr(rag_exp.results, "max_combinations", len(evaluations_list)) or 24

    rag_patterns.metadata["name"] = "rag_patterns_artifact"
    rag_patterns.metadata["uri"] = rag_patterns.uri
    rag_patterns.metadata["metadata"] = {"patterns": []}
    for i, eval in enumerate(evaluations_list):
        patt_dir = rag_patterns_dir / eval.pattern_name
        patt_dir.mkdir(parents=True, exist_ok=True)

        pattern_data = _build_pattern_json(eval, iteration=i, max_combinations=max_combinations)

        generate_notebook_from_templates(
            INDEXING_CELLS_TEMPLATES,
            pattern_data,
            Path(patt_dir, "indexing.ipynb"),
            input_data_key=input_data_key,
        )

        generate_notebook_from_templates(
            GENERATION_CELLS_TEMPLATES,
            pattern_data,
            Path(patt_dir, "inference.ipynb"),
            test_data_key=test_data_key,
        )

        # Flat schema: scores = per-metric aggregates (mean, ci_low, ci_high); final_score
        pattern_data["scores"] = (getattr(eval, "scores", None) or {}).get("scores") or {}
        pattern_data["final_score"] = getattr(eval, "final_score", None)
        rag_patterns.metadata["metadata"]["patterns"].append(pattern_data)
        with (patt_dir / "pattern.json").open("w+", encoding="utf-8") as pattern_details:
            json_dump(pattern_data, pattern_details, indent=2)

        eval_data = evaluation_data_list[i] if i < len(evaluation_data_list) else []
        try:
            q_scores = (eval.scores or {}).get("question_scores") or {}
            if q_scores and all(isinstance(q_scores.get(k), dict) for k in q_scores):
                evaluation_result_list = ExperimentResults.create_evaluation_results_json(eval_data, eval)
            else:
                evaluation_result_list = _evaluation_result_fallback(eval_data, eval)
        except (KeyError, TypeError):
            evaluation_result_list = _evaluation_result_fallback(eval_data, eval)
        with (patt_dir / "evaluation_results.json").open("w+", encoding="utf-8") as f:
            json_dump(evaluation_result_list, f, indent=2)

    # TODO autorag_run_artifact


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        rag_templates_optimization,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
