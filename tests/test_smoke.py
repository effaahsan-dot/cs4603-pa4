"""Offline smoke test for the Document Analyst graph (Bonus A test target).

This is the target the Bonus A CI pipeline runs to prove the graph wires up
before any deploy. Fill it in once your nodes are implemented.

Run:  uv run pytest -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_graph_module_imports():
    """Minimal collection guard: the graph module must import cleanly."""
    from agent.graph import build_graph  # noqa: F401

class FakeLLM:
    """Minimal stand-in for a chat LLM — returns a fixed response, no network."""
    def invoke(self, messages, *args, **kwargs):
        class FakeResponse:
            content = "FAKE_LLM_RESPONSE"
            tool_calls = []
        return FakeResponse()

    def bind_tools(self, tools, *args, **kwargs):
        return self


class FakeRetriever:
    """Minimal stand-in for a vector-search retriever — returns fixed docs."""
    def invoke(self, query, *args, **kwargs):
        class FakeDoc:
            page_content = "Net income was $100 million."
            metadata = {"source": "fake_report.pdf", "page": 1}
        return [FakeDoc()]


class FakeTool:
    """Minimal stand-in for an MCP calculation tool."""
    name = "calculate"
    description = "Fake calculator tool for testing"

    def invoke(self, *args, **kwargs):
        return "42"


def test_build_graph_compiles_with_fakes():
    """The graph must build and compile using injected fakes — no real
    Databricks LLM, retriever, or MCP subprocess required."""
    from agent.graph import build_graph

    graph = build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[FakeTool()])
    assert graph is not None
