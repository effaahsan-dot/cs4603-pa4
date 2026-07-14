"""Full Document Analyst graph (Tasks 1.5 + 1.7).

TODO:
  - `load_mcp_tools(server_path=None)`: connect the GIVEN MCP server over stdio
    (see langchain-mcp-adapters) and return its tools.
  - `make_mcp_node(tools, llm)`: execute one calculation step by letting the LLM
    call exactly one MCP tool, then append the result and increment the index.
  - `build_graph(llm=None, retriever=None, tools=None)`: assemble
    planner -> supervisor -> {rag_agent | mcp_tools} -> ... -> synthesizer.
    Inject dependencies so the graph can be unit-tested offline with fakes.
"""

from __future__ import annotations
import asyncio
import json
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph


from agent.planner import make_planner
from agent.prompts import MCP_STEP_PROMPT
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer


_DEFAULT_SERVER_PATH = str(Path(__file__).resolve().parent.parent / "tools" / "mcp_server.py")


def _run_async(coro):
    """Run a coroutine from sync code whether or not a loop is already
    running in this thread (avoids 'asyncio.run() cannot be called from a
    running event loop' inside MLflow's synchronous serving path)."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

def load_mcp_tools(server_path: str | None = None):
    import os
    import sys

    from langchain_mcp_adapters.client import MultiServerMCPClient

    path = server_path or _DEFAULT_SERVER_PATH

    async def _load():
        client = MultiServerMCPClient({
            "analyst": {
                "command": "python",
                "args": [path],
                "transport": "stdio",
            }
        })
        return await client.get_tools()

    old_stdout, old_stderr = sys.stdout, sys.stderr
    devnull = open(os.devnull, "w")
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        return _run_async(_load())
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        devnull.close()

def make_mcp_node(tools, llm):
    
    llm_with_tools = llm.bind_tools(tools)

    def mcp_tools(state: AnalystState) -> dict:
        position = state["current_step_index"]
        step = state["plan"][position]

        response = llm_with_tools.invoke([
            
            SystemMessage(content=MCP_STEP_PROMPT),
            HumanMessage(content=step),
        ])

        calls = getattr(response, "tool_calls", None)
        if not calls:
            result = f"Step {position + 1} ('{step}'): no calculation tool was called."
        else:
            call = calls[0]
            tool = next((t for t in tools if t.name == call["name"]), None)
            if tool is None:
                result = f"Step {position + 1}: model requested unknown tool '{call['name']}'."
            else:
                raw = _run_async(tool.ainvoke(call["args"]))
                result = raw if isinstance(raw, str) else json.dumps(raw)

        return {
            "step_results": state["step_results"] + [result],
            "current_step_index": position + 1,
        }

    return mcp_tools



def build_graph(llm=None, retriever=None, tools=None):
    
    if llm is None:
        from config import get_chat_llm
        llm = get_chat_llm()
    if retriever is None:
        from rag.store import get_retriever
        retriever = get_retriever()
    if tools is None:
        tools = load_mcp_tools()

    builder = StateGraph(AnalystState)
    builder.add_node("planner", make_planner(llm))
    builder.add_node("supervisor", make_supervisor(llm))
    builder.add_node(RAG, make_rag_agent(retriever, llm))
    builder.add_node(MCP, make_mcp_node(tools, llm))
    builder.add_node(SYNTH, make_synthesizer(llm))

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {RAG: RAG, MCP: MCP, SYNTH: SYNTH},
    )

    builder.add_edge(RAG, "supervisor")
    builder.add_edge(MCP, "supervisor")
    builder.add_edge(SYNTH, END)

    return builder.compile()
