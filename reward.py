from typing import Dict, List

def grade_episode(task_def: dict, action_log: List[dict], ticket_states: Dict[str, dict]) -> float:
    expected = task_def.get("expected_resolution", {})
    if not expected:
        return 0.0

    total_ticket_score = 0.0
    num_tickets = len(expected)

    for ticket_id, criteria in expected.items():
        ticket_score = _grade_ticket(ticket_id, criteria, action_log, ticket_states.get(ticket_id, {}))
        total_ticket_score += ticket_score

    final_score = total_ticket_score / num_tickets if num_tickets > 0 else 0.0
    return max(0.0, min(1.0, final_score))

def _grade_ticket(ticket_id: str, criteria: dict, action_log: List[dict], ticket_state: dict) -> float:
    score = 0.0
    ticket_actions = [a for a in action_log if a.get("ticket_id") == ticket_id or a.get("action_type") == "search_kb"]
    action_types_taken = [a["action_type"] for a in ticket_actions]

    required = criteria.get("required_actions", [])
    if required:
        required_met = sum(1 for r in required if r in action_types_taken)
        score += 0.40 * (required_met / len(required))

    preferred = criteria.get("preferred_actions", [])
    if preferred:
        preferred_met = sum(1 for p in preferred if p in action_types_taken)
        score += 0.10 * (preferred_met / len(preferred))

    forbidden = criteria.get("forbidden_actions", [])
    if forbidden:
        violations = sum(1 for f in forbidden if f in action_types_taken)
        score += 0.20 if violations == 0 else -0.15 * violations
    else:
        score += 0.20

    reply_keywords = criteria.get("reply_must_contain_any", [])
    if reply_keywords:
        reply_msgs = [a.get("message", "").lower() for a in ticket_actions if a["action_type"] in ("reply", "ask_info")]
        all_replies = " ".join(reply_msgs)
        keyword_hits = sum(1 for kw in reply_keywords if kw.lower() in all_replies)
        score += 0.15 * min(1.0, keyword_hits / max(1, min(3, len(reply_keywords))))

    tone_keywords = criteria.get("tone_keywords", [])
    if tone_keywords:
        reply_msgs = [a.get("message", "").lower() for a in ticket_actions if a["action_type"] in ("reply", "ask_info")]
        all_replies = " ".join(reply_msgs)
        tone_hits = sum(1 for kw in tone_keywords if kw.lower() in all_replies)
        score += 0.15 * min(1.0, tone_hits / max(1, min(2, len(tone_keywords))))
    else:
        score += 0.15

    if ticket_state.get("status") == "open":
        score -= 0.05

    return max(0.0, min(1.0, score))


def compute_step_reward(action: dict, task_def: dict, ticket_states: Dict[str, dict], action_history: List[dict], step_number: int) -> float:
    action_type = action.get("action_type", "")
    ticket_id = action.get("ticket_id")
    message = (action.get("message") or "").lower()
    expected = task_def.get("expected_resolution", {})

    reward = 0.0

    if action_type == "search_kb":
        reward += 0.05
        query = (action.get("query") or "").lower()
        for tid, criteria in expected.items():
            if any(kw.lower() in query for kw in criteria.get("reply_must_contain_any", [])):
                reward += 0.05
                break

    elif action_type == "lookup_order":
        if ticket_id in expected:
            reward += 0.10 if "lookup_order" in expected[ticket_id].get("required_actions", []) else 0.03

    elif action_type == "reply":
        if ticket_id in expected:
            criteria = expected[ticket_id]
            if "reply" in criteria.get("required_actions", []): reward += 0.10
            
            tone_hits = sum(1 for kw in criteria.get("tone_keywords", []) if kw.lower() in message)
            reward += min(0.10, tone_hits * 0.05)

            content_hits = sum(1 for kw in criteria.get("reply_must_contain_any", []) if kw.lower() in message)
            reward += min(0.10, content_hits * 0.03)

    elif action_type == "ask_info":
        reward += 0.05

    elif action_type in ("refund", "replace", "escalate"):
        if ticket_id in expected:
            criteria = expected[ticket_id]
            if action_type in criteria.get("required_actions", []) or action_type in criteria.get("preferred_actions", []):
                reward += 0.20
            elif action_type in criteria.get("forbidden_actions", []):
                reward -= 0.30 if action_type == "refund" else 0.25
            else:
                reward += 0.05

    elif action_type == "close":
        if ticket_id in expected:
            past_acts = [a["action_type"] for a in action_history if a.get("ticket_id") == ticket_id]
            other_reqs = [r for r in expected[ticket_id].get("required_actions", []) if r != "close"]
            
            if all(r in past_acts for r in other_reqs) and other_reqs:
                reward += 0.30
            elif not past_acts:
                reward -= 0.40
            else:
                reward += 0.05

    if action_type not in ("reply", "refund", "replace", "escalate", "ask_info", "lookup_order", "search_kb", "close"):
        reward -= 0.10

    if len(action_history) >= 3:
        if all(a.get("action_type") == action_type and a.get("ticket_id") == ticket_id for a in action_history[-3:]):
            reward -= 0.15

    return max(-1.0, min(1.0, reward))
