from pathlib import Path
from typing import TextIO

from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from langchain_core.documents import Document


def load_search_space_from(file: TextIO | Path) -> AI4RAGSearchSpace:
    """Load a search space from a .yml file.

    The .yml must conform to search_space_prep_report_schema. Returns an AI4RAGSearchSpace instance.

    Args:
        file: Path or file-like object to the .yml defining the search space.

    Returns:
        AI4RAGSearchSpace instance.
    """
    return AI4RAGSearchSpace(
        [
            Parameter("inference_model_id", "C", values=["mistral", "llama"]),
            Parameter("embedding_model", "C", values=["mistral_emv", "llama_emb", "granite"]),
            Parameter(
                "retrieval",
                "C",
                values=[
                    {"method": "simple", "window_size": "0", "number_of_chunks": "5"},
                    {"method": "window", "window_size": "2", "number_of_chunks": "8"},
                ],
            ),
        ]
    )


def load_as_langchain_doc(path: str | Path) -> list[Document]:
    """Load text file(s) into a list of langchain Document objects.

    Args:
        path: Local path to a text file or folder of text files.

    Returns:
        List of langchain Document objects.
    """
    if isinstance(path, str):
        path = Path(path)

    documents = []
    if path.is_dir():
        for doc_path in path.iterdir():
            with doc_path.open("r", encoding="utf-8") as doc:
                documents.append(Document(page_content=doc.read(), metadata={"file_name": doc_path.name}))

    elif path.is_file():
        with path.open("r", encoding="utf-8") as doc:
            documents.append(Document(page_content=doc.read(), metadata={"file_name": path.name}))

    return documents
