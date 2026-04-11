import random
from dataclasses import dataclass
from reward import calculate_reward


@dataclass
class CustomerSupportAction:
    action: str


@dataclass
class CustomerSupportObservation:
    ticket_type: str
    customer_sentiment: int
    issue_severity: int
    wait_time_hours: int


@dataclass
class StepResult:
    observation: CustomerSupportObservation
    reward: float
    done: bool


class CustomerSupportEnv:
    """
    OpenEnv Reinforcement Learning Environment for Customer Support Triage.
    """
    def __init__(self, task_level="easy"):
        """
        Initialize the environment.
        :param task_level: "easy", "medium", or "hard"
        """
        self.task_level = task_level
        self.current_state = None
        self.actions = ["resolve", "escalate", "investigate"]
        self.reset()

    @classmethod
    async def from_docker_image(cls, image_name, task_level="hard"):
        """Create environment from a Docker image (OpenEnv compatible)."""
        return cls(task_level=task_level)

    def _generate_ticket(self):
        """Generate random realistic customer support ticket data."""
        ticket_types = ["order_status", "refund_request", "product_complaint", "technical_issue", "billing_error"]
        self.current_state = {
            "ticket_type": random.choice(ticket_types),
            "customer_sentiment": random.randint(1, 10),
            "issue_severity": random.randint(1, 10),
            "wait_time_hours": random.randint(0, 72)
        }

    def reset(self):
        """Reset and generate first ticket."""
        self._generate_ticket()
        return StepResult(
            observation=CustomerSupportObservation(**self.current_state),
            reward=0.0,
            done=False
        )

    def state(self):
        """Return current state as dict."""
        return self.current_state

    def step(self, action):
        """
        Process the agent's action and return StepResult.
        Accepts either a CustomerSupportAction or a plain string.
        """
        if isinstance(action, CustomerSupportAction):
            action_str = action.action
        else:
            action_str = action

        if action_str not in self.actions:
            raise ValueError(f"Invalid action: {action_str}. Must be one of {self.actions}")

        reward = calculate_reward(self.current_state, action_str, self.task_level)

        # Generate next ticket (auto-transition)
        self._generate_ticket()

        return StepResult(
            observation=CustomerSupportObservation(**self.current_state),
            reward=reward,
            done=False
        )

    async def close(self):
        """Cleanup (OpenEnv compatible)."""
        pass
