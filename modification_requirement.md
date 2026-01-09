# PB2S Modification Requirements
## "Industrial Grade" Neuro-Symbolic Architecture Specification

### 1. Executive Summary
This document dictates the technical requirements to implement the **PB2S (Prompt-Based 2-System)** architecture. The system transforms from a static LLM trainer into a dynamic **Neuro-Symbolic Reasoning Engine** where a Recurrent Neural Network (System 1) is trained via reinforcement from a deterministic Symbolic Stack Server (System 2).

### 2. Infrastructure & Orchestration
**Current Gap**: The `trainer` script assumes a connection to `localhost:9000`, but the Docker Orchestrator might be running on `8000` or using outdated code. The "Stack Server" code exists in `services/stack_server` but needs to be the active `orchestrator` service.

**Requirement 2.1: Docker Unification**
- **Action**: Update `infra/docker/docker-compose.yml`.
- **Change**: The `orchestrator` service must build from `services/stack_server` (not `services/orchestrator`).
- **Network**: Ensure the `trainer` container (or local process) can reach `http://localhost:9000` (mapped port).

### 3. "The Brain" - Recurrent Reasoning Network
**Current Gap**: The file `services/model/reasoning_network.py` is missing/deleted. The trainer cannot import the custom architecture.

**Requirement 3.1: Re-Implementation of Universal Transformer**
- **File**: Re-create `services/model/reasoning_network.py`.
- **Specs**:
    - **Class**: `RecurrentReasoningNetwork(nn.Module)`.
    - **Topology**: Universal Transformer (Weights Shared across T steps).
    - **Gating**: `FeedbackGate` (Linear layer) to inject gradient signals between recurrence loops.
    - **Forward Pass**: Must accept an optional `feedback_tensor` to alter the latent state during the "Reflection" phase.

### 4. "The Teacher" - Stack Server (System 2)
**Current Gap**: The `stack_server` code exists (`api.py`, `cae.py`, `irq.py`) and appears robust (SymPy, NLI, Grammar), but it must be verified as the active listener.

**Requirement 4.1: Endpoint Verification**
- **Endpoint**: `POST /audit`
- **Input**: `{"draft_text": str}`
- **Output**: `{"passed": bool, "score": float, "reflection_prompt": str, "dqc_snapshot": {"beta": float}}`
- **Logic**:
    - **CAE**: Runs SymPy (Math), NLI (Logic), LanguageTool (Grammar).
    - **IRQ**: Translates violations (e.g., "MathError") into natural prompts ("Draft failed math. Revise 1+1.").
    - **DQC**: Adjusts `Beta` (Penalty) based on sliding window performance.

### 5. "The Loop" - Trainer Logic
**Current Gap**: `train_drrl.py` needs to orchestration the interactions between the "Brain" and "Teacher".

**Requirement 5.1: Training Loop Logic**
The training step must implementation this exact pseudocode:
```python
# Phase 1: Draft
draft_logits = model(prompt)
draft_text = decode(draft_logits)

# Phase 2: Audit (External World Check)
audit_response = requests.post("http://localhost:9000/audit", json={"draft": draft_text})
reflection_prompt = audit_response["reflection_prompt"]
beta = audit_response["dqc_snapshot"]["beta"]
score = audit_response["score"]

# Phase 3: Revision (Internal Correction)
# Inject the reflection as context OR a feedback embedding
revision_input = prompt + "\nREFLECTION: " + reflection_prompt
revision_logits = model(revision_input)

# Phase 4: Loss Calculation
loss_ce = CrossEntropy(revision_logits, ground_truth)
loss_rule = (1.0 - score) * beta  # DQC regulated penalty
total_loss = loss_ce + loss_rule
```

**Requirement 5.2: Feedback Injection**
- The trainer must tokenize the `reflection_prompt` and append it to the context window so the model "hears" the teacher's correction before revising.

### 6. Dynamic Scaling
**Current Gap**: The `beta` penalty is currently static or non-existent in the loop.

**Requirement 6.1: Adaptive Difficulty**
- The trainer must log the `score` back to the Stack Server (or the Stack Server tracks it automatically via `/audit`) so the `DQC` increments difficulty.
- As the model improves (Score > 0.9 avg), `Beta` should increase (finer tolerance) and Curriculum Difficulty should ramp up.

This specification outlines a "Closed Loop" Neuro-Symbolic system where the Neural Network is subservient to the Symbolic Logic Engine during training.
