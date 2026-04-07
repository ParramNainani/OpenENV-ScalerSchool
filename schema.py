import os
from pydantic import BaseModel
from typing import Literal, Optional, List, Dict

class Ticket(BaseModel):
    id: str
    content: str
    status: Literal["open", "escalated", "replied", "closed"]
    priority: Literal["low", "medium", "high", "critical"]

class Action(BaseModel):
    action_type: Literal["read_ticket", "search_kb", "reply", "escalate", "close"]
    ticket_id: Optional[str] = None
    query: Optional[str] = None
    message: Optional[str] = None

class Observation(BaseModel):
    result: str
    open_tickets: List[str]
    error: Optional[str] = None

class Reward(BaseModel):
    value: float
    reason: str

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, str]

class TaskDef(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str

class InferenceRequest(BaseModel):
    task_id: Literal["easy", "medium", "hard"] = "easy"
    base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4"
    api_key: str = ""

class InferenceResult(BaseModel):
    task_id: str
    total_steps: int
    total_reward: float
    log: List[Dict]
