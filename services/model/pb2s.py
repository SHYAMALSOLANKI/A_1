import os
import torch
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
    2. REFLECT: Send to Backend for CAE/IRQ analysis (Teacher grades + explains only).
    3. REVISE: Generate new thought conditioning on backend feedback.
    4. LEARN:  Update on the delta improvement between Draft and Revision.
    """
    
    def __init__(self, model: RecurrentReasoningNetwork, tokenizer, backend_url=None, optimizer=None, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.backend_url = backend_url or os.getenv("STACK_SERVER_URL", "http://localhost:8000")
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

    def reflect(self, draft: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Stage 2: The Backend (System 2) audits the draft.
        Returns the full audit bundle including IRQ feedback.
        """
        try:
            resp = requests.post(
                f"{self.backend_url}/audit",
                json={"draft_text": draft, "metadata": {"prompt": prompt}},
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
            "reflection_prompt": "REFLECT:\n- System Error: Auditor unreachable.\nREVISE:\nProvide a clear response.",
            "learned_rule": "Always respond clearly even when the auditor is unavailable.",
            "dqc_snapshot": {"beta": 0.1}
        }

    def revise(self, prompt: str, draft: str, feedback: str) -> str:
        """
        Stage 3: Generate a revised version based on feedback.
        """
        feedback_block = self._ensure_reflect_block(feedback)
        composite = (
            f"PROMPT:\n{prompt}\n\n"
            f"DRAFT:\n{draft}\n\n"
            f"{feedback_block}\n"
        )
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

    def learn(self, prompt, draft, feedback, revision, draft_score, revision_score, beta):
        """
        Stage 4: Perform backward pass.
        """
        self.model.train()
        if self.optimizer:
            self.optimizer.zero_grad()
        
        # Target: The model SHOULD have produced the revision given the prompt+feedback
        # Ideally, we also want the model to eventually produce good drafts without feedback,
        # but initially we train the correction mechanism.
        
        feedback_block = self._ensure_reflect_block(feedback)
        composite = (
            f"PROMPT:\n{prompt}\n\n"
            f"DRAFT:\n{draft}\n\n"
            f"{feedback_block}\n"
        )
        target_text = composite + revision
        inputs = self.tokenizer(
            target_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(self.device)
        prefix_ids = self.tokenizer(
            composite,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).input_ids.to(self.device)
        labels = inputs.input_ids.clone()
        labels[:, :prefix_ids.shape[1]] = -100

        outputs = self.model(input_ids=inputs.input_ids, labels=labels)
        ce_loss = outputs["loss"]
        
        # Rule Reward (RL Signal) based on improvement delta
        improvement = max(0.0, revision_score - draft_score)
        rule_penalty = (1.0 - improvement) * beta
        
        total_loss = ce_loss + rule_penalty
        
        if self.optimizer:
            total_loss.backward()
            self.optimizer.step()
        
        return total_loss.item()


    def run_cycle(self, prompt, max_retries=2, min_score=0.8):
        """
        Executes the full Draft->Reflect->Revise loop.
        """
        draft = ""
        for _ in range(max_retries + 1):
            draft = self.draft(prompt)
            if draft.strip():
                break

        feedback = "REFLECT:\n- No feedback available.\nREVISE:\nProvide a clearer response."
        learned_rule = ""
        draft_score = 0.0
        
        # Simple loop: Just one revision step for training efficiency usually, 
        # but we can do retry logic here
        
        audit_data = self.reflect(draft, prompt=prompt)
        draft_score = audit_data.get("score", 0.0)
        feedback = audit_data.get("reflection_prompt", feedback)
        learned_rule = audit_data.get("learned_rule", "")
        if not feedback.strip() or not learned_rule.strip():
            for _ in range(max_retries):
                audit_data = self.reflect(draft, prompt=prompt)
                feedback = audit_data.get("reflection_prompt", feedback)
                learned_rule = audit_data.get("learned_rule", learned_rule)
                if feedback.strip() and learned_rule.strip():
                    break
        beta = float(audit_data.get("dqc_snapshot", {}).get("beta", 0.1))
        
        revision = draft
        if draft_score < min_score:
            for _ in range(max_retries + 1):
                revision = self.revise(prompt, draft, feedback)
                if revision.strip():
                    break
            revision_audit = self.reflect(revision, prompt=prompt)
            revision_score = revision_audit.get("score", draft_score)
        else:
            revision = draft
            revision_score = draft_score
            
        loss = self.learn(prompt, draft, feedback, revision, draft_score, revision_score, beta)

        reflect_section = self._extract_reflect_section(feedback)
        trace_text = (
            f"DRAFT:\n{draft}\n\n"
            f"REFLECT:\n{reflect_section}\n\n"
            f"REVISE:\n{revision}\n\n"
            f"LEARNED:\n{learned_rule}"
        )

        self._store_memory(learned_rule, prompt)
        self._store_trace(prompt, draft, feedback, revision, learned_rule, trace_text, draft_score, revision_score)
            
        return {
            "prompt": prompt,
            "draft": draft,
            "feedback": feedback,
            "revision": revision,
            "score": revision_score,
            "draft_score": draft_score,
            "revision_score": revision_score,
            "loss": loss,
            "learned": learned_rule,
            "trace_text": trace_text
        }

    @staticmethod
    def _ensure_reflect_block(feedback: str) -> str:
        text = feedback.strip()
        if "REFLECT:" not in text:
            text = f"REFLECT:\n{text}"
        if "REVISE:" not in text:
            text = f"{text}\nREVISE:\n"
        return text

    @staticmethod
    def _extract_reflect_section(feedback: str) -> str:
        if "REFLECT:" in feedback:
            after = feedback.split("REFLECT:", 1)[1]
            if "REVISE:" in after:
                return after.split("REVISE:", 1)[0].strip()
            return after.strip()
        return feedback.strip()

    def _store_memory(self, learned_rule: str, prompt: str) -> None:
        if not learned_rule.strip():
            return
        try:
            requests.post(
                f"{self.backend_url}/memory/add",
                json={"learned": learned_rule, "metadata": {"prompt": prompt}},
                timeout=5
            )
        except Exception as e:
            logger.error(f"Memory store error: {e}")

    def _store_trace(
        self,
        prompt: str,
        draft: str,
        feedback: str,
        revision: str,
        learned_rule: str,
        trace_text: str,
        draft_score: float,
        revision_score: float
    ) -> None:
        try:
            requests.post(
                f"{self.backend_url}/trace/store",
                json={
                    "prompt": prompt,
                    "draft": draft,
                    "reflect": feedback,
                    "revise": revision,
                    "learned": learned_rule,
                    "scores": {
                        "draft_score": draft_score,
                        "revision_score": revision_score
                    },
                    "metadata": {"trace": trace_text}
                },
                timeout=5
            )
        except Exception as e:
            logger.error(f"Trace store error: {e}")
