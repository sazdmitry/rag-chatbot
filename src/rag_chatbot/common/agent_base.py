from __future__ import annotations

import time
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from rag_chatbot.models import get_llm
from rag_chatbot.common.prompt_registry import registry


class AgentState(TypedDict, total=False):
    """State passed between agents."""

    next: str
    output: str
    current_session: List[AIMessage]
    history: List[AIMessage]


schema: Dict[str, Any] = {
    "name": "AgentStep",
    "input_schema": {
        "type": "object",
        "properties": {
            "next": {
                "type": "string",
                "description": "The name of the next agent to call",
            },
            "output": {
                "type": "string",
                "description": "The content/output of this agent",
            },
        },
        "required": ["next", "output"],
        "additionalProperties": False,
    },
}

base_llm = get_llm("llama3.2:3b")
llm = base_llm.with_structured_output(schema)
N_RETRY = 5


def call_with_retry(chain, state: AgentState, n_retry: int, name: str):
    """Invoke a chain with simple retry logic."""
    for attempt in range(1, n_retry + 1):
        try:
            return chain.invoke(state)
        except Exception:
            if attempt == n_retry:
                raise
            time.sleep(0.5 * attempt)


class AgentBase:
    def __init__(self, name: str, *prompts: Any):
        self.name = name
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", registry[name]), *prompts]
        )
        self.chain = self.prompt | llm

    def __call__(self, state: AgentState) -> AgentState:
        result = call_with_retry(self.chain, state, N_RETRY, self.name)
        self.process_result(state, result)
        return {**state, **result}

    def process_result(self, state: AgentState, result: AgentState) -> None:
        self.save_output(result, state)

    def save_output(self, result: AgentState, state: AgentState) -> None:
        self.save_to_current_session(state, result["output"])

    def save_to_current_session(self, state: AgentState, message: str) -> None:
        state.setdefault("current_session", []).append(AIMessage(message))

    def save_to_history(self, state: AgentState, message: str) -> None:
        state.setdefault("history", []).append(AIMessage(message))
