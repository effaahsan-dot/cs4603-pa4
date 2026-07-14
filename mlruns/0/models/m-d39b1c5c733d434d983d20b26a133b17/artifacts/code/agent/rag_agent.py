"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search.

TODO: Implement `make_rag_agent(retriever, llm)` returning a node that:
  - retrieves top-k chunks for the current step,
  - formats them with [source: file, p.N] citations,
  - extracts a single cited fact via the LLM (or 'not found in documents'),
  - appends the fact to step_results and increments current_step_index.
Reuse `rag/store.py::get_retriever()` so local and deployed retrieval match.
"""

from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage
from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState


def format_docs(docs) -> str:
    if not docs:
        return ""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[source: {source}, p.{page}]\n{doc.page_content}")
    return "\n\n".join(parts)




def make_rag_agent(retriever, llm):
    def rag_agent(state: AnalystState) -> dict:
        index = state["current_step_index"]

        step = state["plan"][index]

        docs = retriever.invoke(step)

        if not docs:
            result = "not found in documents"
        else:
            context = format_docs(docs)
            response = llm.invoke([
                SystemMessage(content=RAG_EXTRACT_PROMPT),
                HumanMessage(content=f"Question: {step}\n\nRetrieved context:\n{context}"),
            ])
            result = response.content

        return {
            "step_results": state["step_results"] + [result],
            "current_step_index": index + 1,
        }
    return rag_agent
