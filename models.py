from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class Action(BaseModel):
    action_type: str
    ticket_id: Optional[str] = None
    message: Optional[str] = None
    query: Optional[str] = None
    reason: Optional[str] = None

class Ticket(BaseModel):
    ticket_id: str
    customer_name: str
    customer_message: str
    personality: str
    order_id: Optional[str] = None
    category: str
    priority: str
    status: str = "open"
    interaction_history: List[Dict[str, Any]] = Field(default_factory=list)
    resolution_notes: Optional[str] = None

class OrderItem(BaseModel):
    product_id: str
    product_name: str
    quantity: int
    price: float

class Order(BaseModel):
    order_id: str
    customer_name: str
    items: List[OrderItem]
    total: float
    status: str
    tracking_number: Optional[str] = None
    days_since_order: int
    delivery_expected_days: int

class Observation(BaseModel):
    result: str
    active_tickets: List[str]
    current_ticket: Optional[Ticket] = None
    order_details: Optional[Order] = None
    kb_results: Optional[List[str]] = None
    customer_sentiment: Optional[str] = None
    error: Optional[str] = None

class Reward(BaseModel):
    value: float
    reason: str

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, str]

class EnvironmentState(BaseModel):
    task_id: str
    difficulty: str
    step_count: int
    max_steps: int
    tickets: Dict[str, Ticket]
    orders: Dict[str, Order]
    cumulative_reward: float
    done: bool

class InferenceRequest(BaseModel):
    task_id: str = "easy"
    base_url: str = "https://api.groq.com/openai/v1"
    model_name: str = "llama-3.3-70b-versatile"
    api_key: Optional[str] = None

class InferenceResult(BaseModel):
    task_id: str
    total_steps: int
    total_reward: float
    final_score: float
    log: List[Dict[str, Any]]
