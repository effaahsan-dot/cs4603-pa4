"""Synthesizer node (Task 1.6).

TODO: Implement `make_synthesizer(llm)` returning a node that combines
step_results into one cited answer and writes it to BOTH `final_answer` AND
the `messages` channel as an AIMessage (required for the OpenAI-compatible
serving contract — see spec Task 1.6).
"""


from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState

#Takes everything gathered in step_results and writes one final answer.
#Writes it to both final_answer and messages (as an AIMessage) — the
#deployed endpoint reads its response from the last message, not from
#final_answer, so skipping the AIMessage means an empty reply

def make_synthesizer(llm):

    def synthesizer(state: AnalystState) -> dict:
        
        question = state["messages"][0].content
        results = state["step_results"]

        if results:
            results_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(results))
        else:
            results_text = "(nothing was gathered)"

        prompt = f"Question: {question}\n\nWhat we found:\n{results_text}"

        answer = llm.invoke([
            SystemMessage(content=SYNTHESIZER_PROMPT),
            HumanMessage(content=prompt),
        ]).content

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    return synthesizer

  
