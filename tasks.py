import copy
from typing import Dict, Any

from models import Ticket

ALL_TASKS = {
    "easy": {
        "difficulty": "easy",
        "max_steps": 8,
        "tickets": {
            "TKT-001": Ticket(
                ticket_id="TKT-001",
                customer_name="Sarah Mitchell",
                customer_message="Hi there! I placed an order a few days ago and I was wondering if you could tell me where it is? My order number is ORD-50123. Thank you so much!",
                personality="polite",
                order_id="ORD-50123",
                category="order_status",
                priority="low",
            )
        },
        "expected_resolution": {
            "TKT-001": {
                "required_actions": ["lookup_order", "reply", "close"],
                "forbidden_actions": ["refund", "replace", "escalate"],
                "reply_must_contain_any": ["tracking", "trk", "shipped", "days"],
                "tone_keywords": ["thank", "help", "happy"],
            }
        }
    },
    "medium": {
        "difficulty": "medium",
        "max_steps": 15,
        "tickets": {
            "TKT-002": Ticket(
                ticket_id="TKT-002",
                customer_name="James Rodriguez",
                customer_message="Hello, I got my package today but it's not what I ordered. I ordered Running Shoes size 10 (ORD-50999) but got a Women's size 8. Please help.",
                personality="confused",
                order_id="ORD-50999",
                category="wrong_item",
                priority="high",
            ),
            "TKT-003": Ticket(
                ticket_id="TKT-003",
                customer_name="Maria Gonzalez",
                customer_message="Hi, I'd like to cancel my order ORD-51234 please. I found the webcam cheaper elsewhere. It hasn't shipped yet right?",
                personality="polite",
                order_id="ORD-51234",
                category="cancellation",
                priority="medium",
            )
        },
        "expected_resolution": {
            "TKT-002": {
                "required_actions": ["replace", "close"],
                "preferred_actions": ["lookup_order", "search_kb"],
                "forbidden_actions": ["refund", "escalate"],
                "reply_must_contain_any": ["replace", "shipping", "return", "label"],
                "tone_keywords": ["sorry", "apologize", "understand"],
            },
            "TKT-003": {
                "required_actions": ["refund", "close"],
                "preferred_actions": ["lookup_order"],
                "forbidden_actions": ["replace", "escalate"],
                "reply_must_contain_any": ["cancel", "refund", "processed"],
                "tone_keywords": [],
            }
        }
    },
    "hard": {
        "difficulty": "hard",
        "max_steps": 25,
        "tickets": {
            "TKT-004": Ticket(
                ticket_id="TKT-004",
                customer_name="David Chen",
                customer_message="This is completely unacceptable! My order ORD-60001 is insanely late. And to make matters worse, the Noise-Canceling Earbuds are completely broken out of the box. You owe me a full refund for my $499 order immediately!",
                personality="angry",
                order_id="ORD-60001",
                category="defective_item",
                priority="urgent",
            ),
            "TKT-005": Ticket(
                ticket_id="TKT-005",
                customer_name="Emily Carter",
                customer_message="I bought an office chair (ORD-60002) and after 4 days the armrest just snapped off. Can you send me a new one?",
                personality="frustrated",
                order_id="ORD-60002",
                category="damaged_item",
                priority="high",
            ),
            "TKT-006": Ticket(
                ticket_id="TKT-006",
                customer_name="Alex Johnson",
                customer_message="I ordered running shoes (ORD-60003) and they don't fit at all. This is ridiculous sizing. Give me a manager now, I want my money back.",
                personality="angry",
                order_id="ORD-60003",
                category="fit_issue_escalation",
                priority="urgent",
            )
        },
        "expected_resolution": {
            "TKT-004": {
                "required_actions": ["replace", "close"],
                "preferred_actions": ["lookup_order", "search_kb"],
                "forbidden_actions": ["refund"],
                "reply_must_contain_any": ["replace", "earbuds", "photo"],
                "tone_keywords": ["apologize", "sorry", "understand", "frustration"],
            },
            "TKT-005": {
                "required_actions": ["replace", "close"],
                "preferred_actions": ["lookup_order", "search_kb"],
                "forbidden_actions": ["refund", "escalate"],
                "reply_must_contain_any": ["replace", "chair", "shipping"],
                "tone_keywords": ["sorry", "frustrating", "understand"],
            },
            "TKT-006": {
                "required_actions": ["close"],
                "preferred_actions": ["escalate", "reply"],
                "forbidden_actions": ["replace", "refund"],
                "reply_must_contain_any": ["return", "manager", "escalate"],
                "tone_keywords": ["apologize", "sorry", "understand"],
            }
        }
    }
}

def get_task(task_id: str) -> Dict[str, Any]:
    task = ALL_TASKS.get(task_id)
    if not task:
        task = ALL_TASKS["easy"]
    return copy.deepcopy(task)

def load_all_tasks():
    return list(ALL_TASKS.keys())
