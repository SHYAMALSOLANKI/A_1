from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import datetime

# Import components
from pps import PPS
from cae import CAE
from irq import IRQ
from quality import DynamicQualityController

app = FastAPI(title="PB2S Stack Server", version="1.0-alpha")

# Initialize Stacks
pps = PPS()
# DQC must be instantiated first so CAE can use it
dqc = DynamicQualityController() 
cae = CAE(dqc_ref=dqc)
irq = IRQ()

# --- Data Models ---
class AuditRequest(BaseModel):
    draft_text: str
    metadata: Optional[Dict] = {}

class AuditResponse(BaseModel):
    passed: bool
    score: float
    violations: List[Dict]
    reflection_prompt: str
    dqc_snapshot: Dict

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
    audit_result = cae.audit(request.draft_text)
    
    # 2. Update DQC
    for v in audit_result["violations"]:
        dqc.update(v["type"], 0.0) # Fail
    if audit_result["passed"]:
        dqc.update("global", 1.0) # Pass
    else:
        # Partial credit based on score
        dqc.update("global", audit_result["score"])

    # 3. Translate IRQ
    reflection = irq.translate_violations(audit_result["violations"])
    
    return {
        "passed": audit_result["passed"],
        "score": audit_result["score"],
        "violations": audit_result["violations"],
        "reflection_prompt": reflection,
        "dqc_snapshot": dqc.report_metrics()
    }

# --- Standalone Execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
