# PB2S: Prompt-Based 2-System Architecture
## Industrial-Grade Neuro-Symbolic AI System

This repository contains the implementation of **PB2S (Prompt-Based 2-System)**, a specialized AI architecture designed to solve the limitations of standard "System 1" Large Language Models (LLMs) by integrating a rigid "System 2" Symbolic Auditor into the training loop.

Unlike standard models (GPT, Llama, Mistral) which rely on a single probabilistic pass to generate answers, PB2S implements a **Draft-Reflect-Revise-Learn (DRRL)** cycle. It forces the neural network to "think" (via recurrent layers) and "verify" (via external backend calls) before finalizing knowledge.

---

## 1. Concept: The Two-System Approach

Modern LLMs suffer from hallucinations because they are purely probabilistic engines (System 1). They lack a "ground truth" verification mechanism during generation.

**PB2S solves this by decoupling Reasoning from Knowledge:**
*   **System 1 (Neural)**: The "Intuition Engine". A specialized Recurrent Neural Network that generates Drafts.
*   **System 2 (Symbolic)**: The "Auditor". A deterministic backend server that checks drafts against Logic, Math (SymPy), and Grammar rules.

The model is **not** trained to simply predict the next token. It is trained to **minimize the distance between its Draft and a policy-compliant Revision**, effectively learning to "audit itself" over time.

---

## 2. Technical Architecture

The system is composed of distributed services running in a Dockerized environment.

### Artifacts & File Structure

\\\
A_1/
 services/
    model/                  # The Neural Components
       reasoning_network.py # Custom Torch Architecture
       pb2s.py             # Controller Logic
   
    stack_server/           # The System 2 Backend
       api.py              # FastAPI Entrypoint
       cae.py              # Constitutional AI Enforcer (Auditor)
       irq.py              # Interactive Reasoning Query (Reflection)
       pps.py              # Policy Protection System
   
    trainer/                # Training Orchestration
        train_drrl.py       # Main Training Loop
        Dockerfile          # Trainer Environment

 infra/
     docker/
         docker-compose.yml  # Service Mesh configuration
\\\

### Infrastructure Map
*   **Orchestrator (Port 9000)**: Host the \stack_server\. Providing \/audit\, \/policy\, and \/quality\ endpoints.
*   **Trainer**: A Python container that loads the Model and iteratively queries the Orchestrator.
*   **Postgres**: Vector database for long-term memory (optional integration).

---

## 3. Specialized PB2S Model vs. Standard LLMs

The \RecurrentReasoningNetwork\ differs from standard Transformers in fundamental ways:

### A. Architectural Recursion (Universal Transformer)
*   **Standard LLMs (Llama/Mistral)**: Use a fixed stack of layers (e.g., 32 layers). An input passes through each layer exactly once.
*   **PB2S Model**: Uses a "Thinking Core" (e.g., 6 layers) that is **looped M times** (e.g., 3 loops).
    *   The model re-applies the *same weights* to the data multiple times, simulating "pondering".
    *   Effective Depth = \Physical Layers\  \Reasoning Loops\.
    *   This increases reasoning capability without increasing parameter count.

### B. The Controller (\PB2SModel\)
This is not just a PyTorch Module; it is a **Managed Entity**.
*   It does not support standard \.generate()\.
*   It implements \un_cycle(prompt)\, which enforces the rigid lifecycle of generation. It is impossible to generate text without it passing through the audit logic.

---

## 4. Training Stage Operations (DRRL Loop)

The training process is defined in \services/trainer/train_drrl.py\. It is an **Online Learning** system, not a static dataset loader.

### The Algorithm:
1.  **DRAFT**: The Neural Network generates a response to a prompt ($).
    update_files.py \hat{y}_{draft} = f_\\theta(x) update_files.py
2.  **AUDIT**: The \PB2SModel\ sends \\hat{y}_{draft}\ to the **Stack Server** (\http://orchestrator:8000/audit\).
3.  **REFLECT**: The Stack Server runs:
    *   **PPS**: Checks for forbidden content.
    *   **CAE**: Checks logical consistency.
    *   **IRQ**: Generates a "Reflection Prompt" ($) explaining *why* the draft failed (or succeeded).
4.  **REVISE**: The Neural Network generates a new response conditioned on the reflection.
    update_files.py \hat{y}_{revised} = f_\\theta(x, \hat{y}_{draft}, R) update_files.py
5.  **LEARN**: The Optimizer updates weights $\\theta$ to maximize the probability of the *Revised* output, penalizing the *Draft* output errors.

### Backend Implementation (\services/stack_server\)
During training, the backend is **live**.
*   **API**: \POST /audit\ receives the raw text.
*   **Logic**: It does not use another LLM to grade. It uses **Deterministic Rules** (SymPy for math, Python AST for code, Regex/Grammar for language) to provide objective loss signals.

---

## 5. Weight Update Mechanism

The system uses a modified Policy Gradient / Supervised Fine-Tuning hybrid approach handled within \pb2s.learn()\:

1.  **Forward Pass 1**: Draft generation (No Gradients, \	orch.no_grad\).
2.  **External Call**: Latency gap for HTTP request to Auditor.
3.  **Forward Pass 2**: Revision generation (Gradients Enabled).
4.  **Loss Calculation**:
    *   Standard Cross-Entropy Loss on the **Revised** tokens.
    *   Weighted by \(1 - Score)\ received from the auditor.
    *   If the Auditor gives a score of 1.0 (Perfect), the loss is minimal.
    *   If the Auditor gives a score of 0.0 (Fail), the gradients are scaled up to force drastic adaptation.

\\\python
# Conceptual Loss
Loss = CrossEntropy(Prediction, Target) * (1 + (Target_Score - Draft_Score))
\\\

This ensures the model effectively "learns to listen" to the symbolic backend.
