"""Planner node (Task 1.2).

TODO: Implement `make_planner(llm)` returning a node that:
  - reads the user question from state["messages"],
  - asks the LLM (PLANNER_PROMPT) for a JSON list of 2-5 steps,
  - parses it robustly (fallback to a single step on parse failure),
  - returns {"plan": [...], "current_step_index": 0, "step_results": []}.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import PLANNER_PROMPT
from agent.state import AnalystState


def _parse_plan(text: str) -> list[str] | None:
    """Extract a JSON list of strings from LLM output, tolerating markdown
    fences or extra prose around it. Returns None if nothing valid is found."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
        return parsed
    return None


def make_planner(llm):
    def planner(state: AnalystState) -> dict:
        user_question = state["messages"][-1].content

        response = llm.invoke([
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=user_question),
        ])

        plan = _parse_plan(response.content)
        if not plan:
            plan = [user_question]

        return {
            "plan": plan,
            "current_step_index": 0,
            "step_results": [],
        }

    return planner