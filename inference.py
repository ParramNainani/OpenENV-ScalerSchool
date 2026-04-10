import os
import json
import re
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
ENV_URL = os.getenv("OPENENV_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an AI customer support agent for an e-commerce company.
AVAILABLE ACTIONS (exactly ONE JSON object per turn):
1. {"action_type": "search_kb", "query": "..."}
2. {"action_type": "lookup_order", "ticket_id": "TKT-XXX"}
3. {"action_type": "reply", "ticket_id": "TKT-XXX", "message": "..."}
4. {"action_type": "ask_info", "ticket_id": "TKT-XXX", "message": "..."}
5. {"action_type": "refund", "ticket_id": "TKT-XXX"}
6. {"action_type": "replace", "ticket_id": "TKT-XXX"}
7. {"action_type": "escalate", "ticket_id": "TKT-XXX", "reason": "..."}
8. {"action_type": "close", "ticket_id": "TKT-XXX"}

WORKFLOW:
1. Lookup order.
2. Search KB for policies.
3. Reply empathetically.
4. Execute resolution (refund, replace, escalate).
5. Close ticket.
Do NOT issue refunds for items that don't fit (without return) or are simply late/in-transit.
Only respond with a raw JSON object."""

def log_start(task: str): print(f"[START] task={task} env=customer_support_resolution model={MODEL_NAME}")
def log_step(step, action, reward, done, error): print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}")
def log_end(success, steps, score, rewards): print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}")

def llm_choose_action(messages: list) -> dict:
    try:
        res = client.chat.completions.create(model=MODEL_NAME, messages=messages, max_tokens=300, temperature=0.2)
        content = re.sub(r"^```(?:json)?|```$", "", res.choices[0].message.content.strip(), flags=re.IGNORECASE).strip()
        if not content.startswith("{"):
            match = re.search(r'\{[^{}]*\}', content)
            if match: content = match.group()
        return json.loads(content)
    except Exception as e:
        return {"action_type": "search_kb", "query": "help"}

def run_task(task_id: str):
    obs = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id}).json()
    log_start(task_id)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards = []
    steps = 0
    done = False
    
    while not done and steps < 25:
        messages.append({"role": "user", "content": f"Observation:\n{json.dumps(obs, indent=2)}\n\nNext action (JSON only):"})
        action = llm_choose_action(messages)
        messages.append({"role": "assistant", "content": json.dumps(action)})
        
        try:
            resp = requests.post(f"{ENV_URL}/step", json=action).json()
        except: break
        
        obs, rew, done = resp["observation"], resp["reward"]["value"], resp["done"]
        rewards.append(rew)
        steps += 1
        log_step(steps, json.dumps(action), rew, done, obs.get("error"))

    score = max(0.01, min(0.99, sum(rewards)))
    log_end(success=score > 0.1, steps=steps, score=score, rewards=rewards)

if __name__ == "__main__":
    for t in ["easy", "medium", "hard"]:
        run_task(t)
        print("")
