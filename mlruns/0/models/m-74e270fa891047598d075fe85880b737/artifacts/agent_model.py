"""MLflow models-from-code definition (Task 2.1).

TODO: Make this file self-contained so MLflow can serialise it:
  - validate DATABRICKS_HOST/TOKEN/MODEL at import time (clear error if missing),
  - rebuild the graph with production clients (LLM, Vector Search retriever,
    MCP tools),
  - end with `mlflow.models.set_model(graph)`.

Must import cleanly:  python -c "import deployment.agent_model"
"""

from __future__ import annotations

# TODO: import os, mlflow, build_graph, get_chat_llm, get_retriever, load_mcp_tools
import os

import mlflow

from agent.graph import build_graph, load_mcp_tools
from config import get_chat_llm

from rag.store import get_retriever
# TODO: validate env vars
_REQUIRED_ENV_VARS = ["DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_MODEL"]
_missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
if _missing:
    raise EnvironmentError(
        f"Missing required environment variables for deployment: {', '.join(_missing)}. "
        "Set these as environment_vars on the serving endpoint (see Task 2.2)."
    )


# TODO: graph = build_graph(...)
llm1 = get_chat_llm()
retriever1 = get_retriever()
tools1 = load_mcp_tools()

# TODO: 
graph = build_graph(llm=llm1, retriever=retriever1, tools=tools1)
mlflow.models.set_model(graph)