"""Python client SDK for the deployed Document Analyst (Part 3)."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import openai


class AnalystClientError(Exception):
    def __init__(self, message: str, status_code=None, request_id=None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class DocumentAnalystClient:
    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.endpoint_name = endpoint_name
        self.host = (host or os.environ.get("DATABRICKS_HOST", "")).rstrip("/")
        self.token = token or os.environ.get("DATABRICKS_TOKEN", "")

        if not self.host or not self.token:
            raise AnalystClientError(
                "DATABRICKS_HOST and DATABRICKS_TOKEN must be passed in or "
                "set as environment variables."
            )

        self.timeout = timeout
        self.max_retries = max_retries
        self._client = openai.OpenAI(
            api_key=self.token,
            base_url=f"{self.host}/serving-endpoints",
            timeout=timeout,
        )

    def _call_with_retry(self, fn, *args, **kwargs):
        start = time.monotonic()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except openai.APITimeoutError as e:
                elapsed = time.monotonic() - start
                raise TimeoutError(
                    f"Request to '{self.endpoint_name}' timed out after {elapsed:.1f}s"
                ) from e
            except openai.APIStatusError as e:
                last_error = e
                if e.status_code in (429, 503) and attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise AnalystClientError(
                    str(e),
                    status_code=e.status_code,
                    request_id=getattr(e, "request_id", None),
                ) from e

        raise AnalystClientError(str(last_error))

    def ask(self, question: str) -> str:
        response = self._call_with_retry(
            self._client.chat.completions.create,
            model=self.endpoint_name,
            messages=[{"role": "user", "content": question}],
        )
        state = response[0]
        return state.final_answer

    def ask_streaming(self, question: str) -> Iterator[str]:
        yield self.ask(question)

    def health_check(self) -> bool:
        try:
            from databricks.sdk import WorkspaceClient

            w = WorkspaceClient(host=self.host, token=self.token)
            ep = w.serving_endpoints.get(self.endpoint_name)
            return str(ep.state.ready) == "EndpointStateReady.READY"
        except Exception:
            return False