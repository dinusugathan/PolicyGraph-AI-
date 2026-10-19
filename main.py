import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from engine.graph import policy_graph_app, retriever
from engine.observability import get_observability_status

app = FastAPI(
    title="Akbar Travels PolicyGraph AI",
    description="Enterprise Multi-Agent Policy Support & Dispute Resolution System with LangChain, LangGraph & LangSmith",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str = Field(..., description="User query or dispute message")
    channel: str = Field("B2C", description="Customer channel: 'B2C', 'B2B', or 'ALL'")

class ChatResponse(BaseModel):
    query: str
    channel: str
    intent: str
    response: str
    citations: List[str]
    escalated: bool
    escalation_reason: Optional[str] = None
    refund_calculation: Optional[Dict[str, Any]] = None
    guardrail_passed: bool
    trace_steps: List[Dict[str, Any]]

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    initial_state = {
        "query": req.query,
        "channel": req.channel.upper(),
        "trace_steps": []
    }
    
    result = policy_graph_app.invoke(initial_state)
    
    return ChatResponse(
        query=result["query"],
        channel=result["channel"],
        intent=result["intent"],
        response=result["response"],
        citations=result.get("citations", []),
        escalated=result.get("escalated", False),
        escalation_reason=result.get("escalation_reason"),
        refund_calculation=result.get("refund_calculation"),
        guardrail_passed=result.get("guardrail_passed", True),
        trace_steps=result.get("trace_steps", [])
    )

@app.get("/api/policies")
async def list_policies():
    """Lists all indexed Akbar Travels policies available in the vector store."""
    return [
        {
            "id": doc.metadata["id"],
            "category": doc.metadata["category"],
            "channel": doc.metadata["channel"],
            "title": doc.metadata["title"],
            "content": doc.page_content
        }
        for doc in retriever.documents
    ]

@app.get("/api/observability")
async def observability_endpoint():
    """Returns LangSmith observability and tracing status."""
    return get_observability_status()

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": "Akbar Travels PolicyGraph AI",
        "graph_nodes": ["router_node", "retriever_node", "policy_agent_node", "refund_calculator_node", "human_escalation_node", "guardrail_node"],
        "total_policies_indexed": len(retriever.documents)
    }

# Mount static web directory
web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(web_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8006, reload=True)
