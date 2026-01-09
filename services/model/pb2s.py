import torch
import torch.nn as nn
import requests
import logging
from typing import Dict, Any, Optional

from services.model.reasoning_network import RecurrentReasoningNetwork

logger = logging.getLogger(__name__)

class PB2SModel:
    """
    The PB2S (Prompt-Based 2-System) Controller.
    
    This class wraps the Recurrent Neural Network (System 1) and manages the 
    conversation with the Stack Server (System 2).
    
    It implements the 4-stage lifecycle:
    1. DRAFT:  Generate initial thought.
    2. REFLECT: Send to Backend for CAE/IRQ analysis.
    3. REVISE: Generate new thought conditioning on backend feedback.
    4. LEARN:  Backpropagate on the delta between Draft and Revision.
    """
    
    def __init__(self, model: RecurrentReasoningNetwork, tokenizer, backend_url="http://localhost:9000", optimizer=None, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.backend_url = backend_url
        self.optimizer = optimizer
        self.device = device
        # Use simpler budget defaults if env not set
        self.budget = {
            "draft_max": 128,
            "revise_max": 256
        }

    def draft(self, prompt: str) -> str:
        """
        Stage 1: Generate a first draft.
        """
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Manual generation loop for Recurrent Network control
        generated_ids = inputs.input_ids
        with torch.no_grad():
            for _ in range(self.budget["draft_max"]): 
                outputs = self.model(input_ids=generated_ids)
                next_token = torch.argmax(outputs["logits"][:, -1, :], dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # Robust stripping
        if full_text.startswith(prompt):
            return full_text[len(prompt):].strip()
        return full_text

    def reflect(self, draft: str) -> Dict[str, Any]:
        """
        Stage 2: The Backend (System 2) audits the draft.
        Returns the full audit bundle including IRQ feedback.
        """
        try:
            resp = requests.post(
                f"{self.backend_url}/audit",
                json={"draft_text": draft},
                timeout=5
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(f"Backend Audit Failed: {resp.status_code}")
        except Exception as e:
            logger.error(f"Backend Connection Error: {e}")
            
        return {
            "passed": False,
            "score": 0.0,
            "reflection_prompt": "System Error: Auditor unreachable.",
            "dqc_snapshot": {"beta": 0.1}
        }

    def revise(self, prompt: str, feedback: str) -> str:
        """
        Stage 3: Generate a revised version based on feedback.
        """
        composite = f"{prompt}\nREFLECTION: {feedback}\nREVISE: "
        inputs = self.tokenizer(composite, return_tensors="pt").to(self.device)
        
        generated_ids = inputs.input_ids
        with torch.no_grad():
             for _ in range(self.budget["revise_max"]):
                outputs = self.model(input_ids=generated_ids)
                next_token = torch.argmax(outputs["logits"][:, -1, :], dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        full_rev = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # Strip context to get pure revision
        # Note: composite length in chars is rough, better do token stripping, 
        # but for string return this works if unique
        if full_rev.startswith(composite):
            return full_rev[len(composite):].strip()
        return full_rev

    def learn(self, prompt, draft, revision, score, beta):
        """
        Stage 4: Perform backward pass.
        """
        self.model.train()
        if self.optimizer:
            self.optimizer.zero_grad()
        
        # Target: The model SHOULD have produced the revision given the prompt+feedback
        # Ideally, we also want the model to eventually produce good drafts without feedback,
        # but initially we train the correction mechanism.
        
        # Self-Supervised Target: The Revision itself is the ground truth
        inputs = self.tokenizer(
            revision, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding="max_length"
        ).to(self.device)
        
        outputs = self.model(input_ids=inputs.input_ids, labels=inputs.input_ids)
        ce_loss = outputs["loss"]
        
        # Rule Penalty (RL Signal)
        rule_penalty = (1.0 - score) * beta
        
        total_loss = ce_loss + rule_penalty
        
        if self.optimizer:
            total_loss.backward()
            self.optimizer.step()
        
        return total_loss.item()


    def run_cycle(self, prompt, max_retries=3, min_score=0.8):
        """
        Executes the full Draft->Reflect->Revise loop.
        """
        draft = self.draft(prompt)
        feedback = "Initial"
        score = 0.0
        
        # Simple loop: Just one revision step for training efficiency usually, 
        # but we can do retry logic here
        
        audit_data = self.reflect(draft)
        score = audit_data.get("score", 0.0)
        feedback = audit_data.get("reflection_prompt", "")
        beta = float(audit_data.get("dqc_snapshot", {}).get("beta", 0.1))
        
        if score < min_score:
            revision = self.revise(prompt, feedback)
            # Re-Audit revision? Ideally yes, but for Training Step we just learn from it
        else:
            revision = draft # It was good!
            
        loss = self.learn(prompt, draft, revision, score, beta)
            
        return {
            "prompt": prompt,
            "draft": draft,
            "feedback": feedback,
            "revision": revision,
            "score": score,
            "loss": loss
        }
