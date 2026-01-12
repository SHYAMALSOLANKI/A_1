import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn

from services.stack_server.pps import PPS
from services.stack_server.cae import CAE
from services.stack_server.irq import IRQ
from services.stack_server.quality import DynamicQualityController
from services.stack_server.memory_store import MemoryStore
from services.stack_server.trace_store import TraceStore
from services.stack_server.truth import TruthTool

app = FastAPI(title="PB2S Stack Server", version="1.0-alpha")

# Initialize Stacks
pps = PPS()
# DQC must be instantiated first so CAE can use it
dqc = DynamicQualityController()
truth_tool = TruthTool(
    enabled=os.getenv("ENABLE_WEB_SOT", "false").lower() == "true"
)
cae = CAE(
    dqc_ref=dqc,
    truth_tool=truth_tool,
    config={
        "enable_nli": os.getenv("ENABLE_NLI", "false").lower() == "true",
        "enable_grammar": os.getenv("ENABLE_GRAMMAR", "true").lower() == "true",
    }
)
irq = IRQ()
memory_store = MemoryStore()
trace_store = TraceStore()

# --- Data Models ---
class AuditRequest(BaseModel):
    draft_text: str
    metadata: Optional[Dict] = {}

class AuditResponse(BaseModel):
    passed: bool
    score: float
    violations: List[Dict]
    severity: int
    gate_breakdown: List[Dict]
    subscores: Dict[str, float]
    reflection_prompt: str
    learned_rule: Optional[str]
    dqc_snapshot: Dict

class TraceRequest(BaseModel):
    prompt: str
    draft: str
    reflect: str
    revise: str
    learned: str
    scores: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class MemoryAddRequest(BaseModel):
    learned: str
    metadata: Dict[str, Any] = {}

class MemoryQueryRequest(BaseModel):
    prompt: str
    top_k: int = 3

# --- Routes ---

@app.get("/health")
def health_check():
    return {"status": "active", "system": "PB2S Stack Server"}

@app.get("/config/policy")
def get_policy():
    return pps.policies

@app.get("/quality")
def get_quality_metrics():
    return dqc.report_metrics()

@app.post("/audit", response_model=AuditResponse)
async def audit_draft(request: AuditRequest):
    """
    Main Loop Entry Point for the Trainer.
    1. Audits the Draft (CAE)
    2. Updates DQC with scores
    3. Translates Violations (IRQ)
    4. Returns feedback for the Trainer to inject.
    """
    # 1. Audit
    audit_result = cae.audit(request.draft_text, metadata=request.metadata)
    
    # 2. Update DQC
    if audit_result["passed"]:
        dqc.update("global", audit_result["score"])
    else:
        dqc.update("global", 0.0)
    for v in audit_result["violations"]:
        dqc.update(v["type"], 0.0)

    # 3. Translate IRQ
    reflection = irq.translate_violations(
        audit_result["violations"],
        audit_result["severity"],
        audit_result["gate_breakdown"]
    )
    learned_rule = irq.build_learned_rule(audit_result["violations"])
    
    return {
        "passed": audit_result["passed"],
        "score": audit_result["score"],
        "violations": audit_result["violations"],
        "severity": audit_result["severity"],
        "gate_breakdown": audit_result["gate_breakdown"],
        "subscores": audit_result["subscores"],
        "reflection_prompt": reflection,
        "learned_rule": learned_rule,
        "dqc_snapshot": dqc.report_metrics()
    }

@app.post("/trace/store")
async def store_trace(request: TraceRequest):
    record = {
        "prompt": request.prompt,
        "draft": request.draft,
        "reflect": request.reflect,
        "revise": request.revise,
        "learned": request.learned,
        "scores": request.scores,
        "metadata": request.metadata,
    }
    trace_store.append(record)
    return {"status": "stored"}

@app.post("/memory/add")
async def memory_add(request: MemoryAddRequest):
    stored = memory_store.add(request.learned, request.metadata)
    return {"status": "stored", "record": stored}

@app.post("/memory/query")
async def memory_query(request: MemoryQueryRequest):
    results = memory_store.query(request.prompt, top_k=request.top_k)
    return {"results": results}

# --- Standalone Execution ---
if __name__ == "__main__":
    port = int(os.getenv("STACK_SERVER_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
