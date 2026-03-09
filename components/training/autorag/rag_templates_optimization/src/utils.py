from pathlib import Path
from typing import TextIO

from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from langchain_core.documents import Document


def load_search_space_from(file: TextIO | Path) -> AI4RAGSearchSpace:
    """
    Loads a search space defined in a .yml file.
    The .yml file must conform to the _autorag/search_space_preparation/search_space_prep_report_schema.yml_.

    Args:
      file
          A Path-like or File-like object pointing to the .yml file defining the search space.

    Returns:

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
    """
    Given path to a text-based file or a folder thereof load everything to memory and
    return as a list of langchain `Document` objects.

    Args:
        path
            A local path to either a text file or a folder of text files.
    Returns:
        A list of langchain `Document` objects.

    Note:

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
