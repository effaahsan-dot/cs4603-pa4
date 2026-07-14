"""Supervisor node + routing edge (Task 1.3).

TODO:
  - `make_supervisor(llm)`: if current_step_index >= len(plan) -> next_agent =
    'synthesizer'; else classify the current step to 'rag_agent' or 'mcp_tools'.
  - `route_from_supervisor(state)`: return state["next_agent"] for the
    conditional edge.
"""

from __future__ import annotations

from agent.state import AnalystState
from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import SUPERVISOR_PROMPT

#these are the agents that are getting saved and that will get roted to from the supervisor node

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"


def make_supervisor(llm):
    def supervisor(state: AnalystState) -> dict:
        
        position = state["current_step_index"]
        plan = state["plan"]

        #if current position is greater than the entire plan workflow then we have reached the end and can call on the synthesizer 
        if position >= len(plan):
            return {"next_agent": SYNTH}

        step = plan[position]

        response = llm.invoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=step),
        ])

        route = response.content.strip().lower()
        route = "".join(c for c in route if c.isalnum() or c == "_")

        if route not in (RAG, MCP, SYNTH):
            route = MCP if any(kw in step.lower() for kw in
                                ("calculate", "compute", "growth", "percent", "%")) else RAG

        return {"next_agent": route}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    return state["next_agent"]
