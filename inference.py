"""
Inference Script — AI Customer Support Triage
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

STDOUT FORMAT
- [START] task=<task_name> env=<benchmark> model=<model_name>
- [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
- [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from environment import CustomerSupportAction, CustomerSupportEnv

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = "customer_support_triage"
MAX_STEPS = 5
TEMPERATURE = 0.0
MAX_TOKENS = 10
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are a customer support triage AI. You receive ticket data and must decide
    the optimal action based on ticket type, customer sentiment, issue severity,
    and wait time.
    Choose ONE of: 'resolve', 'escalate', or 'investigate'.
    Reply with exactly one word — the action name, nothing else.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def get_model_action(client: OpenAI, state_info: dict) -> str:
    """Ask the LLM to choose an action given the current ticket state."""
    prompt = (
        f"Ticket data: {state_info}\n"
        f"Choose your action:"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip().lower()
        return text if text in ["resolve", "escalate", "investigate"] else "investigate"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "investigate"


async def run_task(task_name: str, task_level: str) -> None:
    """Run a single task evaluation episode."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await CustomerSupportEnv.from_docker_image(IMAGE_NAME, task_level=task_level)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            state_info = {
                "ticket_type": result.observation.ticket_type,
                "customer_sentiment": result.observation.customer_sentiment,
                "issue_severity": result.observation.issue_severity,
                "wait_time_hours": result.observation.wait_time_hours,
            }

            action = get_model_action(client, state_info)
            result = env.step(CustomerSupportAction(action=action))

            reward = result.reward
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done, error=error)

            if done:
                break

        # Score = average reward, clamped to [0, 1]
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    tasks = [
        {"name": "Task1_Easy", "level": "easy"},
        {"name": "Task2_Medium", "level": "medium"},
        {"name": "Task3_Hard", "level": "hard"},
    ]

    for task_info in tasks:
        await run_task(task_info["name"], task_info["level"])


if __name__ == "__main__":
    asyncio.run(main())
