# PB2S (Prompt-Based 2-System) Architecture Documentation

## 1. Executive Summary
This project implements a **Neuro-Symbolic Reasoning Engine** that learns from scratch (Tabula Rasa). 
Unlike traditional LLMs that learn by **Imitation** (predicting the next word from a dataset), this architecture learns by **Reflection** (correcting its own errors based on feedback from a rigid logic engine).

---

## 2. Core Artifacts

### A. The "Brain" (Neural Network)
*   **Type**: Deep & Narrow Transformer (Custom Llama Config).
*   **Size**: ~58 Million Parameters.
*   **Topology**: 
    *   **Vocabulary**: 8,192 (Tiny). Forces the model to understand concepts rather than memorizing rare words.
    *   **Layers**: 16 (Deep). Reasoning requires sequential computation steps.
    *   **Width**: 512 (Narrow). Prevents "hiding" memorized facts in the residual stream.
*   **Role**: To generate creative drafts and adaptive revisions.

### B. The "Mind" (Tokenizer)
*   **Type**: Custom Byte-Level BPE.
*   **Training Data**: `training_data/curriculum.jsonl`.
*   **Features**:
    *   **Digit-Splitting**: Ensures `1984` is tokenized as `1`, `9`, `8`, `4` (or similar atomic units) to enable math.
    *   **Control Tokens**: `DRAFT:`, `REFLECT:`, `REVISE:` are first-class citizens in the vocabulary.

### C. The "Teacher" (Stack Server)
*   **Location**: `localhost:9000`.
*   **Components**:
    *   **SymPy**: Validates math equation truth.
    *   **LanguageTool**: Enforces grammar and structure.
    *   **NLI (Cross-Encoder)**: Enforces logical consistency between statements.
    *   **DQC**: Dynamic Quality Controller that ramps up difficulty as the model learns.

---

## 3. Methodology of Operation (The "Thought Loop")

The system operates in a **Recurrent Cycle** for every single prompt.

1.  **Draft**: The Neural Net takes the User Query and predicts until it hits a stop token.
    *   *Result*: A raw, potentially hallucinated answer.
2.  **Audit**: The Stack Server intercepts the Draft. It runs deterministic code (Python) to check facts and logic.
    *   *Result*: A list of errors (e.g., "Line 3 Contradicts Line 1").
3.  **Reflect**: The Neural Net receives the Error List. It generates a "Reflection" (a plan to fix it).
4.  **Revise**: The Neural Net executes the plan and generates the Final Answer.

---

## 4. Methodology of Training

We do not use a static dataset of "Correct Answers". We use a dataset of **Questions**.

**The Algorithm:**
1.  **Forward Pass 1**: Model tries to answer a Question.
2.  **Intervention**: Server scores the attempt. If bad, it forces a Revision (using the Feedback as a prompt).
3.  **Forward Pass 2**: Model generates the Revision.
4.  **Outcome**:
    *   If Revision > Draft (Improvement): We accept the Revision as a "synthetic label".
    *   We train the model to **predict the Revision** given the (Prompt + Draft + Feedback).
5.  **Effect**: The model learns the *function* $f(Draft, Feedback) \rightarrow Improved\_Draft$.
    *   Eventually, it learns to do this internally without the external server (Internalization of the Inner Critic).

---

## 5. Execution Roadmap

1.  **Artifact Creation**:
    *   Train Custom Tokenizer on `curriculum.jsonl`.
    *   Define `drrl_config.yaml` with the Deep/Narrow topology.
2.  **Initialization**:
    *   Instantiate Model with Random Weights.
3.  **Training**:
    *   Run `train_drrl.py` to start the self-correcting loop.
