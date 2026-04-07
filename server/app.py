from fastapi import FastAPI, HTTPException, Body
from .schema import Action, Observation, Reward, StepResponse, Ticket, TaskDef, InferenceRequest, InferenceResult
import copy
from typing import Literal, List, Dict
import json
import re
from openai import OpenAI
import os

app = FastAPI(title="OpenEnv - Customer Support Triage", description="Simulates a real-world customer support queue.")

# Global state
KNOWLEDGE_BASE = {
    "refund": "Refunds can be issued within 30 days of purchase if the item is unused. Use polite tone.",
    "password": "Users can reset passwords via the 'Forgot Password' link on the login page.",
    "server down": "Apologize and inform the user that engineering is investigating the outage (ETA 2 hours)."
}

TASKS = {
    "easy": {
        "tickets": {
            "T1": Ticket(id="T1", content="My server is completely down, everything is offline!!!", status="open", priority="critical"),
        },
        "target": "escalate"
    },
    "medium": {
        "tickets": {
            "T2": Ticket(id="T2", content="I forgot my password.", status="open", priority="low"),
            "T3": Ticket(id="T3", content="Can I get my money back?", status="open", priority="medium")
        },
        "target": "resolve_all"
    },
    "hard": {
        "tickets": {
            "T4": Ticket(id="T4", content="Server down and burning money!", status="open", priority="critical"),
            "T5": Ticket(id="T5", content="Forgot password again", status="open", priority="low"),
            "T6": Ticket(id="T6", content="Need refund for unused software", status="open", priority="medium"),
        },
        "target": "full_triage"
    }
}

current_task = "easy"
current_state = copy.deepcopy(TASKS["easy"])
steps_taken = 0
MAX_STEPS = 15

@app.get("/")
def ping():
    return {"status": "ok"}

@app.get("/state")
def get_state():
    return {"tickets": {k: v.dict() for k,v in current_state.get("tickets", {}).items()}, "task": current_task}

@app.post("/reset", response_model=Observation)
def reset(task_id: str = "easy"):
    global current_state, current_task, steps_taken
    if task_id not in TASKS:
        task_id = "easy"
    current_task = task_id
    current_state = copy.deepcopy(TASKS[task_id])
    steps_taken = 0
    return Observation(
        result=f"Environment reset to task: {task_id}", 
        open_tickets=list(current_state["tickets"].keys())
    )

@app.post("/step", response_model=StepResponse, summary="Take an environment step", description="Executes an action in the support triage environment and advances the state.")
def step(
    action: Action = Body(
        ...,
        openapi_examples={
            "read_ticket": {
                "summary": "Read Ticket",
                "value": {"action_type": "read_ticket", "ticket_id": "T1"}
            },
            "search_kb": {
                "summary": "Search KB",
                "value": {"action_type": "search_kb", "query": "password reset"}
            },
            "reply": {
                "summary": "Reply to Customer",
                "value": {"action_type": "reply", "ticket_id": "T1", "message": "Here is how to reset your password..."}
            },
            "escalate": {
                "summary": "Escalate Ticket",
                "value": {"action_type": "escalate", "ticket_id": "T1"}
            },
            "close": {
                "summary": "Close Ticket",
                "value": {"action_type": "close", "ticket_id": "T1"}
            }
        }
    )
):
    global steps_taken, current_state
    steps_taken += 1
    
    observation = Observation(result="", open_tickets=[k for k, v in current_state["tickets"].items() if v.status == "open"])
    reward = Reward(value=0.0, reason="No progress")
    done = False
    
    try:
        if action.action_type == "search_kb":
            res = [v for k, v in KNOWLEDGE_BASE.items() if action.query and action.query.lower() in k.lower()]
            observation.result = f"KB Search Results: {res}" if res else "No results found."
            reward = Reward(value=0.05, reason="Searched KB (partial progress)")
        
        elif action.action_type in ["read_ticket", "reply", "escalate", "close"]:
            if not action.ticket_id or action.ticket_id not in current_state["tickets"]:
                observation.error = "Invalid or missing ticket_id."
                reward = Reward(value=-0.1, reason="Invalid action format.")
            else:
                ticket = current_state["tickets"][action.ticket_id]
                
                if action.action_type == "read_ticket":
                    observation.result = f"Content: {ticket.content} | Priority: {ticket.priority}"
                
                elif action.action_type == "reply":
                    ticket.status = "replied"
                    observation.result = f"Sent reply to {ticket.id}."
                    reward = Reward(value=0.2, reason="Replied to ticket")
                
                elif action.action_type == "escalate":
                    ticket.status = "escalated"
                    observation.result = f"Escalated {ticket.id}."
                    reward = Reward(value=0.2, reason="Escalated ticket")
                    
                elif action.action_type == "close":
                    if ticket.status == "open":
                        reward = Reward(value=-0.5, reason="Closed without taking action!!!")
                        observation.result = f"Closed {ticket.id} prematurely."
                    else:
                        ticket.status = "closed"
                        observation.result = f"Successfully closed {ticket.id}."
                        reward = Reward(value=0.5, reason="Ticket resolved and closed correctly.")
    except Exception as e:
        observation.error = str(e)
        reward = Reward(value=-0.2, reason="Exception during step")
        
    # Grading (simplified for brevity)
    open_t = sum(1 for t in current_state["tickets"].values() if t.status == "open")
    if open_t == 0:
        done = True
        reward = Reward(value=1.0, reason="All tickets triaged/resolved!")
        
    if steps_taken >= MAX_STEPS:
        done = True
        reward = Reward(value=0.0, reason="Max steps reached.")
        
    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
        info={"steps": str(steps_taken)}
    )

@app.post("/run_inference", response_model=InferenceResult, summary="Run automated LLM inference", tags=["Inference"])
def run_inference(req: InferenceRequest):
    # Retrieve the API key
    api_key = req.api_key or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing API Key. Provide it in the request or set HF_TOKEN environment variable.")
    
    client = OpenAI(base_url=req.base_url, api_key=api_key)
    
    def llm_choose_action(msgs):
        try:
            res = client.chat.completions.create(
                model=req.model_name,
                messages=msgs,
                max_tokens=150
            )
            content = res.choices[0].message.content.strip()
            content = re.sub(r"^```(json)?", "", content, flags=re.IGNORECASE).strip()
            content = re.sub(r"```$", "", content).strip()
            return json.loads(content)
        except Exception as e:
            return {"action_type": "search_kb", "query": f"fallback_error: {str(e)}"}

    obs = reset(task_id=req.task_id)
    # Re-encode to dict to match client-side logic
    obs_dict = {"result": obs.result, "open_tickets": obs.open_tickets, "error": obs.error}
    
    done = False
    step_num = 0
    total_reward = 0.0
    execution_log = []

    system_prompt = """You are an AI customer support agent resolving a queue of tickets. 
Your goal is to clear the 'open_tickets' list by taking actions on them.
For each ticket you MUST:
1. read_ticket to understand it.
2. If unsure, search_kb for policy.
3. reply to or escalate the ticket to resolve it.
4. close the ticket exactly once after resolving it.

Available actions (Return exactly one as a raw JSON object):
{"action_type": "search_kb", "query": "YOUR_QUERY"}
{"action_type": "read_ticket", "ticket_id": "T1"}
{"action_type": "reply", "ticket_id": "T1", "message": "YOUR_MSG"}
{"action_type": "escalate", "ticket_id": "T1"}
{"action_type": "close", "ticket_id": "T1"}

Respond ONLY with valid JSON. No markdown, no conversational text."""

    messages = [{"role": "system", "content": system_prompt}]

    while not done and step_num < 15:
        messages.append({"role": "user", "content": f"Observation: {json.dumps(obs_dict)}\nWhat is your next action JSON?"})
        
        action_dict = llm_choose_action(messages)
        messages.append({"role": "assistant", "content": json.dumps(action_dict)})
        
        # Build action properly (filter out extra keys internally)
        action_obj = Action(**action_dict)
        
        step_response = step(action=action_obj)
        obs_dict = {
            "result": step_response.observation.result, 
            "open_tickets": step_response.observation.open_tickets, 
            "error": step_response.observation.error
        }
        rew = step_response.reward.value
        done = step_response.done
        total_reward += rew
        
        execution_log.append({
            "step": step_num,
            "action": action_dict,
            "observation": obs_dict,
            "reward": rew,
            "done": done
        })
        step_num += 1

    return InferenceResult(
        task_id=req.task_id,
        total_steps=step_num,
        total_reward=total_reward,
        log=execution_log
    )

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
