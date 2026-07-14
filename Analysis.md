# CS4603 PA4 — Document Analyst

> This `README.md` is a **graded deliverable**:
>
> - Document how to set up, run, and deploy your Document Analyst so a TA can reproduce your results.
> - **Answer every ANALYSIS QUESTION** from the assignment in the sections below.
> - Code that runs but is not explained will not receive full marks.
> - Replace every `TODO` before submitting.
> - Keep it self-contained: a reader should be able to follow this file top-to-bottom —
>   setup → ingest → run → deploy → results — without opening the assignment PDF.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your values
```

## Running locally

TODO: how to ingest the corpus, run the graph in `pa4.ipynb`, and test queries.

> **Example of the level of detail expected** (replace with your own steps/values):
>
> 1. **Ingest the corpus** (run once, from a Databricks notebook):
>    ```python
>    from rag.ingest import ingest
>    ingest(spark, volume_path="/Volumes/main/default/pa4/annual_report.pdf")
>    ```
>    This parses the PDF, chunks it into `main.default.ali_analyst_chunks`, and syncs the
>    Vector Search index `main.default.ali_analyst_index`. Wait until the index is `READY`.
>
> 2. **Build and run the graph** in `pa4.ipynb`:
>    ```python
>    from agent.graph import build_graph
>    graph = build_graph()          # uses config.py + rag/store.py + the MCP server
>    result = graph.invoke({"messages": [{"role": "user",
>              "content": "What was the net revenue in 2023?"}]})
>    print(result["messages"][-1].content)
>    ```
>
> 3. **Test queries I ran** (retrieval-only, computation-only, combined):
>    | Query | Answer produced |
>    |-------|-----------------|
>    | "What was the net income in 2023?" | ¥1.11 trillion [source: annual_report.pdf, p.4] |
>    | "What is 15% of 2.4 billion?" | 360 million |
>    | "What was 2023 revenue, and its value after 10% growth?" | ¥16.91T → ¥18.60T (16.91 × 1.10) |

## Deployment

TODO: how you logged, registered, and served the model; endpoint name; URL.

## Design decisions

TODO: graph architecture, routing, deployment choices.

---

## Analysis Questions

### Task 1.2 — Planner
1. It's not handled automatically. Each node only sees `plan[current_step_index]` 
   not the results from earlier steps so a step like "then compound it" would have
   no idea what "it" refers to. The planner prompt works around this by instructing
   self-contained steps (spelling out numbers explicitly) instead of relying on the
   graph to resolve dependencies at execution time. Only the synthesizer at the very
   end sees all step results together.

2. Not worth it unconditionally extra LLM call per step for latency/cost with little
   benefit on simple two-step queries. It would help narrowly when a lookup fails
   (e.g. "not found in documents") right now the next step just proceeds with bad
   input. A replan triggered only on failure reformulating just that one query 
   would be worth it replanning after every step would not be.

### Task 1.3 — Supervisor
1.  A misroute sends a step to the wrong specialist for example a calculation step routed
   to `rag_agent` searches the document for something that isn't there and comes back
   "not found," which then affects the final answer. Right now nothing detects this
   automatically. A simple fix can be that if `rag_agent` returns "not found," treat that as a
   signal the step may have been misrouted and retry it through `mcp_tools` (or vice
   versa) before giving up.

2. A single ReAct agent with all tools is simpler as there is one loop, no separate planner and supervisor. The supervisor pattern is worth the extra complexity
   when steps are clearly typed (lookup vs. calculation) and you want the plan visible
   and before execution so it is easier to debug and explain than one agent
   silently deciding tool-by-tool. For simple single-tool queries, a plain ReAct agent
   would be faster with no accuracy loss.

### Task 1.4 — RAG Agent
1. Retrieving on the single decomposed step is generally better than the full original
   question. The original question often mixes a lookup intent and a math intent
   ("what was revenue... and what would it be after 8% growth")  which embeds as a
   blurry combination of both and hurts similarity search. A narrow step like "Find
   Meridian's net revenue for fiscal year 2023" is closer to what's actually written in
   the document so it retrieves more precisely as long as the planner scoped the step
   well. If it didn't retrieval quality drops
   .
2. For a vague step like "find relevant financial data," I would rewrite the query before
   retrieving pass the vague step plus the original user question to the LLM and ask it
   to produce one specific, document-shaped search phrase and then retrieve on that instead of the raw step text. 
   This is cheaper than and reduces reliance on LLM to filter out noise and it keeps the
   fix localized to the RAG agent rather than requiring a better planner prompt.

### Task 2.1 — Model Definition
1. `models-from-code` works by shipping `agent_model.py` to a completely separate machine
   (the serving container) and re-running it there from scratch, with nothing carried over
   from wherever it was written. Any external state such as a local database, a notebook
   variable, a fake retriever built in `pa4.ipynb` simply doesn't exist on that machine.
   Referencing it either crashes the import immediately (e.g. "file not found") or, worse,
   fails silently later at request time. That's why this file rebuilds the LLM, retriever,
   and MCP tools fresh inside itself instead of assuming anything from earlier in the
   assignment is still around.

2. Querying the managed Vector Search index live means the corpus can be updated (new
   documents, corrections to existing ones) without redeploying the model at all, and the
   container image stays small since it holds no embedded data. The cost is a network
   dependency on every request such as added latency, and if vector search is down or slow the
   whole model is degraded even though the LLM itself is fine. Baking the corpus into the
   artifact removes that network dependency and could be faster per-request but freezes the data at deploy time any update will require full re-deployment which is expensive, time-consuming 
   and loses the benefit of a managed shared service other models could also query.

### Task 2.3 — Serving Endpoint
1. Being allowed to run the model and being able to call the LLM are two different
   things. Databricks' own auth just means "you're allowed to deploy this." But inside
   the code `get_chat_llm()` makes its own separate HTTP call out to the LLM endpoint 
   and that call needs its own token to prove who's asking, the same way any API client
   would. No token there, no way for the model to actually talk to the LLM, even though
   the endpoint itself is running fine.

2. Databricks keeps the old version running until the new one is fully ready, then
   switches traffic over  so nothing in progress gets cut off. The catch is you're
   paying for both versions at once during that window, which is the tradeoff for not
   dropping any requests.


### Task 3.2 — Client
1. Why is exponential backoff better than fixed-interval retries for a model serving endpoint?

The reason why exponential backoff is better is that incase the endpoint is overloaded already with multiple requests and is unable to process them in an easier manner/way. Having a fixed-interval will cause the requests to be made all over again after the same time interval not facilitating the overloaded endpoint in any way. However, incase an exponential backoff is taken into consideration, the requests tend to spread out over time facilitating the endpoint and giving it space to process and breathe as well.

2. Your client has a max_retries parameter. What is the danger of setting it too high in a production system with many concurrent users?

Again, if the endpoint is already overloaded with multiple requests coming from different users increasing the max_retries simply delays clarity for the user and they keep on waiting for a simple reply or result way longer as the process of degradation is going to take longer. The systemd does not degrade and end in time, efficiently and is a burden on the compute resources for no reason.

3. When would you choose ask_streaming() over ask()? Give a concrete UX example.

ask_streaming() is the right choice whenever the user is watching a live response and perceived latency matters more than total completion time for example a chat UI where showing the answer word by word as it's generated feels responsive versus a blank screen for several seconds followed by the whole paragraph appearing at once.

### Bonus A / B / C (if attempted)
TODO
