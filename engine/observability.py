import os
import time
from typing import Dict, Any, Callable
from dotenv import load_dotenv

load_dotenv()

LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true" and bool(os.getenv("LANGCHAIN_API_KEY"))
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "akbar-travels-policygraph")

def get_observability_status() -> Dict[str, Any]:
    """Returns the current LangSmith tracing status and active configuration."""
    return {
        "langsmith_active": LANGSMITH_ENABLED,
        "project": LANGSMITH_PROJECT,
        "tracing_v2": os.getenv("LANGCHAIN_TRACING_V2", "false"),
        "api_key_configured": bool(os.getenv("LANGCHAIN_API_KEY")),
        "endpoint": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    }

def record_node_trace(node_name: str, input_summary: str, output_summary: str, duration_ms: float) -> Dict[str, Any]:
    """Records a structured trace span for a graph node."""
    return {
        "node": node_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_ms": round(duration_ms, 2),
        "input_summary": input_summary,
        "output_summary": output_summary,
        "langsmith_traced": LANGSMITH_ENABLED
    }
