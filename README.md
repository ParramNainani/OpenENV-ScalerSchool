---
title: OpenENV - AI Customer Support Resolution
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🛒 AI Customer Support Resolution Environment (OpenEnv)

A real-world **OpenEnv** environment simulating e-commerce customer support interactions. Agents must understand customer intent, consult internal policies, handle emotional customers with empathy, and take correct resolution actions — all while avoiding common traps like unnecessary refunds or premature ticket closures.

## 🌟 Why This Environment?

Customer support is one of the most impactful real-world applications of AI agents. This environment goes beyond simple chatbot Q&A — it requires:

- **Multi-step reasoning**: Agents must gather information before acting
- **Policy compliance**: Actions must align with company refund, replacement, and escalation policies
- **Emotional intelligence**: Angry and frustrated customers require empathetic tone before solutions
- **Trap avoidance**: Deliberately placed scenarios where the naive action (e.g., full refund) is wrong
- **Multi-ticket management**: Medium and Hard tasks require managing multiple tickets simultaneously

---

## 🧠 Observation Space

```json
{
  "result": "string — what happened after the last action",
  "active_tickets": ["TKT-001", "TKT-002"],
  "current_ticket": { "ticket_id": "...", "customer_name": "...", "customer_message": "...", "personality": "angry|confused|polite|frustrated", "status": "..." },
  "order_details": { "order_id": "...", "items": [...], "total": 399.97, "status": "shipped", "tracking_number": "TRK-..." },
  "kb_results": ["[Refund Policy]: Refunds available within 30 days..."],
  "customer_sentiment": "satisfied — customer is calming down",
  "error": "null"
}
```

## ⚙️ Action Space (8 Actions)

| Action | Description | Required Fields |
|--------|-------------|-----------------|
| `search_kb` | Search internal knowledge base | `query` |
| `lookup_order` | Retrieve order details for a ticket | `ticket_id` |
| `reply` | Send a response to the customer | `ticket_id`, `message` |
| `ask_info` | Request more information from customer | `ticket_id`, `message` |
| `refund` | Issue a refund for the order | `ticket_id` |
| `replace` | Issue a replacement for wrong/defective items | `ticket_id` |
| `escalate` | Escalate to human supervisor | `ticket_id`, `reason` (optional) |
| `close` | Close the ticket as resolved | `ticket_id` |

```json
{"action_type": "reply", "ticket_id": "TKT-001", "message": "I'm sorry for the inconvenience..."}
```

---

## 🧪 Tasks (Easy → Medium → Hard)

### 🟢 Easy — Order Status Inquiry
**Customer**: Sarah (polite) — "Where is my order?"  
**Expected**: Look up order → Reply with tracking info → Close  
**Trap**: Do NOT refund/escalate — order is within delivery window  

### 🟡 Medium — Wrong Item + Cancellation (2 tickets)
**Customer 1**: James (confused) — Received wrong shoe size  
**Customer 2**: Maria (polite) — Wants to cancel unshipped order  
**Expected**: KB lookup → Replacement for James, Refund for Maria → Close both  
**Trap**: Do NOT refund wrong-item tickets (offer replacement instead)  

### 🔴 Hard — Multi-Issue Angry Customers (3 tickets)
**Customer 1**: David (angry 😡) — Late delivery + defective earbuds, demands $495 full refund  
**Customer 2**: Emily (frustrated 😤) — Damaged chair, wants replacement  
**Customer 3**: Alex (angry 😡) — Shoes don't fit, demands manager  
**Expected**: Policy-compliant resolutions with empathetic tone  
**Traps**:
- David: Only earbuds ($200) are defective, not the whole $495 order → Replace earbuds + shipping discount
- Alex: Shoes don't fit ≠ defective → Offer return, NOT immediate refund

---

## 🏆 Reward Function

### Dense Reward Shaping (per-step signals)
| Action | Reward |
|--------|--------|
| Relevant KB search | +0.05 to +0.10 |
| Order lookup (required) | +0.10 |
| Empathetic reply with correct info | +0.10 to +0.30 |
| Correct refund/replacement | +0.20 |
| Proper escalation | +0.20 |
| Correct ticket closure | +0.30 |

### Penalties
| Violation | Penalty |
|-----------|---------|
| Wrong refund (trap!) | -0.30 |
| Unnecessary escalation | -0.25 |
| Closing without action | -0.40 |
| Repeating same action (loop) | -0.15 |

### Episode Scoring (0.0–1.0)
Final score is graded across 6 dimensions per ticket:
- **Required actions completed** (40%)
- **Forbidden actions avoided** (20%)
- **Reply content quality** (15%)
- **Tone/empathy quality** (15%)
- **Preferred actions taken** (10%)

---

## 📦 Project Structure

```
├── openenv.yaml           ← Environment manifest
├── Dockerfile             ← Docker deployment (port 7860)
├── requirements.txt       ← Python dependencies
├── pyproject.toml         ← Package metadata
├── inference.py           ← Baseline LLM inference script
├── .env.example           ← Environment variable template
├── README.md              ← This file
└── server/
    ├── __init__.py
    ├── app.py             ← FastAPI server (/reset, /step, /state)
    ├── schema.py          ← Pydantic typed models
    ├── environment.py     ← Core environment logic
    ├── tasks.py           ← Task definitions (easy/medium/hard)
    ├── grader.py          ← Deterministic grading engine
    └── knowledge_base.py  ← Simulated KB with policies & order DB
```

---

## 🚀 Setup & Usage

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference (in another terminal)
python inference.py
```

### Docker
```bash
docker build -t customer-support-env .
docker run -p 7860:7860 customer-support-env
```

### Environment Variables
```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your_token_here"
```

---

## 🔗 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/reset?task_id=easy` | Reset environment |
| POST | `/step` | Execute action |
| GET | `/state` | Get internal state |
| POST | `/run_inference` | Run LLM agent |
| GET | `/docs` | Swagger UI |

---

## 📊 Baseline Scores

| Task | Steps | Score | Notes |
|------|-------|-------|-------|
| Easy | 3-4 | 0.70-0.90 | Lookup → Reply → Close |
| Medium | 8-12 | 0.50-0.75 | Multiple tickets, KB required |
| Hard | 15-22 | 0.30-0.55 | Edge cases, traps, emotional handling |

*Scores measured with `llama-3.3-70b-versatile` via Groq API.*