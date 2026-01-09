from pydantic import BaseModel
from typing import Optional, List, Dict

class ChatRequest(BaseModel):
    user_input: str
    conversation_id: int
    turn_order: int
    idempotency_key: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: int
    turn_order: int
    draft: str
    reflection: str
    revision: str
    learned_rule: Optional[str]
    rule_links: List[int]
    status: str
