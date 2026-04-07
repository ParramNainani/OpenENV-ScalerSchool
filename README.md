---
title: OpenENV ScalerSchool
emoji: 🚀
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
---

# OpenEnv: Customer Support Triage

A real-world `OpenEnv` environment simulating a tier-1 customer support inbox. Agents must triage incoming tickets, search the internal KB, formulate responses, escalate critical issues, and close resolved tickets.

## Features
- **Real-World Domain**: Not a toy or grid world. Simulates business logic of a ticketing system.
- **Pydantic Types**: Strict boundaries on Action & Observation spaces (`schema.py`).
- **Graded Tasks**: Easy (single escalation), Medium (KB resolution), Hard (mixed queue).
- **FastAPI Backend**: Fully serves the `step()`, `reset()`, `state()` interface as an HTTP app.
- **Hugging Face Ready**: Built-in `Dockerfile` listening on port `7860`.

## Setup & Deployment (Docker)
```bash
docker build -t openenv_support .
docker run -p 7860:7860 openenv_support
```
_The server will start at `http://localhost:7860`._

## Running the Baseline Inference
Our baseline uses the `openai` Python SDK (configurable via environment variables) to solve the tasks.

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"
export HF_TOKEN="<YOUR_KEY>" # Maps to OpenAI token in standard use
python inference.py
```

## Task Difficulties & Rewards
*   **Easy**: Triaging a single high-priority ticket.
*   **Medium**: Managing multiple tickets with a need to search the KB for correct policies.
*   **Hard**: A combination of priorities and ticket types; heavy penalties for closing a ticket without proper action.
*Reward Function*: Rewards are scaled between `0.0` and `1.0`. Provides partial rewards for KB lookups (`+0.05`), responding (`+0.2`), and resolving completely (`+0.5`), encouraging step-by-step reasoning over random guessing.
`