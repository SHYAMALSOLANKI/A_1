# PB2A Documentation

## Non-Commercial & Ethics Policy
**THIS SOFTWARE IS FOR PERSONAL, NON-COMMERCIAL USE ONLY.**

1. **No Corporate Use**: You may not use this system for corporate intelligence, employee surveillance, or commercial automation.
2. **No Manipulation**: You may not use this system to generate deceptive content, spam, or political influence campaigns.
3. **Data Sovereignty**: This system is designed to run 100% offline. You own your data.

## Configuration & Usage

### Starting Policies
You can select a policy profile by setting the `PB2A_POLICY_PROFILE` environment variable (default: `minimal`).
Available profiles:
- `minimal`: Focus on safety and format.
- `logic_strict`: Prioritize potential contradictions and strict logic.
- `research`: Ask for missing constraints and provide citations.

### Token Budgeting
Configure token headers via env vars:
- `PB2A_CONTEXT_TOKENS`: Total context limit (e.g. 4096).
- `PB2A_DRAFT_RATIO`: Ratio for draft (e.g. 0.20).
- `PB2A_REFLECT_RATIO`: Ratio for reflection (e.g. 0.25).
- `PB2A_REVISE_RATIO`: Ratio for revision (e.g. 0.50).

## Architecture
See `architecture.puml` for a visual overview.
We follow the PB2S methodology:
1. **P**rocess (DRAFT)
2. **B**reakdown (REFLECT)
3. **2** (REVISE)
4. **S**ythesize (LEARN)

## Threat Model
1. **Prompt Injection**: Mitigated by distinct system prompt headers and delimiters.
2. **Bad Learned Rules**: Mitigated by keyword filters and manual admin review.
3. **PII Leakage**: Mitigated by regex-based redaction before database persistence.

## Runbook: Training on Vast.ai
1. Rent an instance with >24GB VRAM (e.g., RTX 3090/4090).
2. SSH into instance.
3. Clone repo.
4. Export dataset: `make export-dataset` (requires connecting to your local DB or uploading a pg_dump).
5. Run: `python -m trainer.train`
6. Download `adapter` artifacts.
