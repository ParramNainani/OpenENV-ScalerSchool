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
