"""LangGraph agent with xAI integration.

This graph implements a conversational agent using xAI's Grok model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import os
from langchain_xai import ChatXAI
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict


class Message(TypedDict):
    """Message format for the conversation."""
    role: str
    content: str


class Context(TypedDict):
    """Context parameters for the agent."""
    model: str
    temperature: float


@dataclass
class State:
    """State for the agent.
    
    Tracks the conversation history and manages the flow of messages.
    """
    messages: List[Message] = field(default_factory=list)
    next_step: str = "call_model"


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input using xAI's model and returns response.
    
    Uses runtime context to configure model parameters.
    """
    # Defensive: runtime.context may be None when invoked from certain runtimes
    ctx = runtime.context or {}

    # Validate messages are present
    if not state.messages:
        raise ValueError(
            "No messages found in the state. Provide at least one human message in state.messages."
        )

    # Read API key from environment and fail-fast with a clear message if missing
    xai_key = os.getenv("XAI_API_KEY")
    if not xai_key:
        raise RuntimeError(
            "XAI_API_KEY environment variable is not set.\n"
            "Set it in your .env file or export it in your shell before running the server."
        )

    # Initialize xAI chat model (read API key from environment variable XAI_API_KEY)
    chat = ChatXAI(
        xai_api_key=xai_key,
        model=ctx.get("model", "grok-4-fast-reasoning"),
        temperature=ctx.get("temperature", 0.7),
    )

    # Generate response (wrap call to surface clearer errors)
    try:
        # Extract textual content from the last message. The incoming message
        # may be a dict with different shapes depending on the caller (for
        # example {'type': 'text', 'text': 'hello'}). Normalize to a string.
        last_msg = state.messages[-1]

        def _extract_text(msg: Any) -> str:
            if isinstance(msg, str):
                return msg
            if isinstance(msg, dict):
                # If the message directly contains 'content', normalize it.
                if "content" in msg:
                    c = msg["content"]
                    if isinstance(c, str):
                        return c
                    if isinstance(c, list):
                        parts: List[str] = []
                        for item in c:
                            parts.append(_extract_text(item))
                        return "\n".join([p for p in parts if p])
                    if isinstance(c, dict):
                        return _extract_text(c)

                # Common alternative keys
                for k in ("text", "message"):
                    if k in msg:
                        return _extract_text(msg[k])

                # Some messages use a type/text shape
                if msg.get("type") == "text" and "text" in msg:
                    return str(msg["text"])

                # Nested 'data' field
                if "data" in msg and isinstance(msg["data"], dict):
                    return _extract_text(msg["data"])

                # Last resort: join primitive values
                vals = []
                for v in msg.values():
                    if isinstance(v, (str, int, float)):
                        vals.append(str(v))
                    else:
                        try:
                            vals.append(_extract_text(v))
                        except Exception:
                            vals.append(str(v))
                return " ".join([v for v in vals if v])

            if isinstance(msg, list):
                return "\n".join([_extract_text(m) for m in msg])

            return str(msg)

        user_text = _extract_text(last_msg)
        response = await chat.ainvoke(user_text)
    except Exception as e:
        # Re-raise with more context for easier debugging in logs
        raise RuntimeError(
            f"xAI model invocation failed: {e}. Last message: {repr(last_msg)}"
        ) from e
    
    # Add response to messages
    state.messages.append({"role": "assistant", "content": response.content})
    
    return {"messages": state.messages, "next_step": "end"}


# Define the graph
graph = StateGraph(State, context_schema=Context)

# Add nodes and configure the graph
graph.add_node("call_model", call_model)
graph.set_entry_point("call_model")
graph.set_finish_point("end")


async def end_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Final no-op end node. Returns the current state and stops the graph."""
    return {"messages": state.messages, "next_step": None}


graph.add_node("end", end_node)

# Compile the graph
graph = graph.compile()

if __name__ == "__main__":
    import asyncio
    
    # Test the graph
    state = State(messages=[{"role": "human", "content": "Tell me about LangGraph"}])
    context = Context(model="grok-4-fast-reasoning", temperature=0.7)
    
    async def test():
        result = await graph.ainvoke(state, context)
        print(result.messages[-1]["content"])
    
    asyncio.run(test())
