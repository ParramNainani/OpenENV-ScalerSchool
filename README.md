---
title: AI Customer Support Triage
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
---

# AI Customer Support Triage & Resolution Environment

## Problem Description
This environment simulates a real-world customer support triage system. An AI agent evaluates incoming support tickets based on ticket type, customer sentiment, issue severity, and wait time to make optimal decisions — resolving simple issues, investigating ambiguous ones, and escalating critical or high-risk situations.

## Environment Explanation
The environment provides randomized, realistic customer ticket data. The state is presented to an AI agent which must return a discrete action.
Depending on the decision quality relative to the specific task difficulty, the environment calculates and returns a reward, representing the agent's performance.

### Observation Space
The state is represented as a dictionary with the following variables:
- `ticket_type`: one of "order_status", "refund_request", "product_complaint", "technical_issue", "billing_error"
- `customer_sentiment`: integer between 1 (calm) and 10 (furious)
- `issue_severity`: integer between 1 (trivial) and 10 (critical)
- `wait_time_hours`: integer between 0 and 72

### Action Space
Discrete text actions:
- `resolve`: Directly resolve the ticket and close it.
- `escalate`: Escalate to a supervisor or specialist.
- `investigate`: Gather more information before taking action.

### Reward Function
Rewards are strictly bound between `0.0` and `1.0`.
- **Correct resolution of simple issue:** `1.0`
- **Risky direct resolution:** `0.2` - `0.5`
- **Resolving a critical issue without care:** `0.0`
- **Correct escalation of critical case:** `1.0`
- **VIP risk detected and escalated:** `1.0`
- **Partial progress:** (e.g., investigating a mid-complexity ticket) returns rewards between `0.3` and `0.5`.

### Tasks Description
- **Task 1 — Easy**: Uses simple rule-based evaluation mapped cleanly from `issue_severity`.
- **Task 2 — Medium**: Applies multiple features including combined urgency from `customer_sentiment` and `issue_severity`, weighing both to determine optimal action.
- **Task 3 — Hard**: Introduces VIP risk detection using `wait_time_hours` alongside high sentiment and severity edge cases requiring careful triage decisions.

## Setup Instructions

### How to Run Locally
1. Clone or download this project.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the baseline agent to see example environment interactions and scores:
```bash
python run_agent.py
```
4. Evaluate all configured tasks with the built-in grading logic:
```bash
python tasks.py
```
5. Execute the automated LLM inference script (required for the Hackathon Submission Portal):
```bash
python inference.py
```

### How to Run with Docker
1. Build the Docker image:
```bash
docker build -t support-env .
```
2. Run the image (Defaults to the baseline agent):
```bash
docker run support-env
```

### How to Deploy to Hugging Face
This repository is pre-configured with a user-friendly Gradio web application explicitly designed for **Hugging Face Spaces**.
1. Navigate to your Hugging Face account and create a new Space.
2. Choose **Gradio** as the Space SDK.
3. Upload all the files inside this repository to your Space.
4. The space will build the environment and automatically map it to the interactive `app.py` script. The UI displays the current state, action interactions, and subsequent reward logic securely!