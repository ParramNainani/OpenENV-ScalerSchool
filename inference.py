import os
import json
import time
import requests
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from .env file securely
load_dotenv()

# Required Env Vars
base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
model_name = os.environ.get("MODEL_NAME", "gpt-4")
api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")

ENV_URL = os.environ.get("OPENENV_URL", "https://instagril-openenv-scalerschool.hf.space")

client = OpenAI(base_url=base_url, api_key=api_key)

def llm_choose_action(messages):
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=150
        )
        content = res.choices[0].message.content.strip()
        # Clean up potential markdown formatting from HF/Groq models
        content = re.sub(r"^```(json)?", "", content, flags=re.IGNORECASE).strip()
        content = re.sub(r"```$", "", content).strip()
        return json.loads(content)
    except Exception as e:
        # fallback action
        return {"action_type": "search_kb", "query": "hello"}

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def run_task(task_id):
    obs = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id}).json()
    done = False
    step = 0
    total_reward = 0.0
    rewards = []
    
    log_start(task=task_id, env="support_triage", model=model_name)

    # Give the agent a memory (message history) so it doesn't repeat itself
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

    while not done and step < 15:
        # Add the current observation to the agent's memory
        messages.append({"role": "user", "content": f"Observation: {json.dumps(obs)}\nWhat is your next action JSON?"})
        
        action = llm_choose_action(messages)
        
        # Add the agent's own action to its memory so it remembers what it did
        messages.append({"role": "assistant", "content": json.dumps(action)})
        
        resp = requests.post(f"{ENV_URL}/step", json=action).json()
        obs = resp["observation"]
        rew = resp["reward"]["value"]
        done = resp["done"]
        total_reward += rew
        rewards.append(rew)
        
        step += 1
        log_step(step=step, action=json.dumps(action), reward=rew, done=done, error=obs.get("error"))
        time.sleep(1)

    success = total_reward > 0
    log_end(success=success, steps=step, score=total_reward, rewards=rewards)

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)
