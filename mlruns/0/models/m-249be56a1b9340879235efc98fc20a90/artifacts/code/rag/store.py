"""Vector Search retriever factory (Task 1.4 support / rag/store.py).

Built directly on `databricks-vectorsearch` rather than `databricks_langchain`,
because `databricks_langchain` pulls in `unitycatalog-langchain[databricks]` ->
`unitycatalog-ai[databricks]` -> `databricks-connect`, and no version of
`databricks-connect` supports Python 3.13 — this broke every deployment build.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import get_settings

TEXT_COLUMN = "chunk_to_retrieve"
CITATION_COLUMNS = ["chunk_id", "source", "page"]


@dataclass
class _SimpleDoc:
    """Minimal stand-in for a LangChain Document — just page_content + metadata,
    which is all agent/rag_agent.py actually reads."""
    page_content: str
    metadata: dict = field(default_factory=dict)


class _VectorSearchRetriever:
    def __init__(self, index, k: int = 4):
        self._index = index
        self._k = k

    def invoke(self, query: str):
        columns = CITATION_COLUMNS + [TEXT_COLUMN]
        results = self._index.similarity_search(
            query_text=query,
            columns=columns,
            num_results=self._k,
        )
        rows = results.get("result", {}).get("data_array", [])
        docs = []
        for row in rows:
            record = dict(zip(columns, row))
            docs.append(_SimpleDoc(
                page_content=record.get(TEXT_COLUMN, ""),
                metadata={
                    "chunk_id": record.get("chunk_id"),
                    "source": record.get("source"),
                    "page": record.get("page"),
                },
            ))
        return docs


def get_vector_store():
    from databricks.vector_search.client import VectorSearchClient

    settings = get_settings()
    vsc = VectorSearchClient()
    return vsc.get_index(settings["vs_endpoint"], settings["vs_index"])


def get_retriever(k: int = 4):
    return _VectorSearchRetriever(get_vector_store(), k=k)