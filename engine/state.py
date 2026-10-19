from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """LangGraph Agent State Schema for Akbar Travels PolicyGraph AI."""
    query: str
    channel: str  # "B2C", "B2B", or "ALL"
    intent: str  # "policy_inquiry", "refund_calculation", "human_escalation", "general_inquiry"
    retrieved_docs: List[Dict[str, Any]]
    refund_inputs: Optional[Dict[str, Any]]
    refund_calculation: Optional[Dict[str, Any]]
    response: str
    citations: List[str]
    escalated: bool
    escalation_reason: Optional[str]
    guardrail_passed: bool
    trace_steps: List[Dict[str, Any]]
