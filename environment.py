import copy
from typing import Dict, List, Optional

from models import Action, Observation, Reward, StepResponse, EnvironmentState
from tasks import get_task
from knowledge_base import search_knowledge_base, lookup_order, ORDER_DATABASE
from reward import compute_step_reward, grade_episode

class CustomerSupportEnvironment:
    def __init__(self):
        self.reset("easy")

    def reset(self, task_id: str = "easy") -> Observation:
        task_def = get_task(task_id)

        self._task_def = task_def
        self._task_id = task_id
        self._difficulty = task_def["difficulty"]
        self._max_steps = task_def["max_steps"]
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._action_log = []

        self._tickets = {tid: (t.model_dump() if hasattr(t, 'model_dump') else t.dict()) for tid, t in task_def["tickets"].items()}
        self._orders = {tdata.get("order_id"): copy.deepcopy(ORDER_DATABASE[tdata.get("order_id")]) for tdata in self._tickets.values() if tdata.get("order_id") in ORDER_DATABASE}

        active_tickets = [tid for tid, t in self._tickets.items() if t["status"] == "open"]
        ticket_summaries = [f"  [{tid}] {t['customer_name']}: \"{t['customer_message'][:80]}...\"" for tid, t in self._tickets.items() if t["status"] == "open"]

        result_text = f"Environment reset — Task: {task_id}\nYou have {len(active_tickets)} active ticket(s):\n" + "\n".join(ticket_summaries)

        return Observation(result=result_text, active_tickets=active_tickets)

    def step(self, action: Action) -> StepResponse:
        if self._done:
            return StepResponse(
                observation=Observation(result="Episode done.", active_tickets=[], error="Episode terminated."),
                reward=Reward(value=0.0, reason="Episode done."),
                done=True,
                info={"step": str(self._step_count)},
            )

        self._step_count += 1
        action_dict = action.model_dump() if hasattr(action, 'model_dump') else action.dict()

        observation = self._process_action(action)
        step_reward = compute_step_reward(action_dict, self._task_def, self._tickets, self._action_log, self._step_count)

        self._action_log.append(action_dict)
        self._cumulative_reward += step_reward

        active_tickets = [tid for tid, t in self._tickets.items() if t["status"] in ("open", "in_progress")]
        all_resolved = len(active_tickets) == 0
        max_steps_reached = self._step_count >= self._max_steps

        if all_resolved or max_steps_reached:
            self._done = True
            final_score = grade_episode(self._task_def, self._action_log, self._tickets)
            if all_resolved:
                blended = 0.3 * step_reward + 0.7 * final_score
                reward = Reward(value=round(max(0.0, min(1.0, blended)), 4), reason=f"Episode complete. Score: {final_score:.3f}")
                observation.result += f"\n\n🎉 All tickets resolved! Score: {final_score:.3f}"
            else:
                reward = Reward(value=round(max(0.0, min(1.0, final_score * 0.5)), 4), reason=f"Max steps. Score: {final_score:.3f}")
                observation.result += f"\n\n⏰ Max steps reached. Score: {final_score:.3f}"
        else:
            reward = Reward(value=round(max(-1.0, min(1.0, step_reward)), 4), reason=self._get_reward_reason(action_dict, step_reward))

        observation.active_tickets = active_tickets
        return StepResponse(observation=observation, reward=reward, done=self._done, info={"step": str(self._step_count), "score": str(self._cumulative_reward)})

    def get_state(self) -> EnvironmentState:
        return EnvironmentState(
            task_id=self._task_id, difficulty=self._difficulty, step_count=self._step_count, max_steps=self._max_steps,
            tickets=self._tickets, orders=self._orders, cumulative_reward=self._cumulative_reward, done=self._done
        )

    def _process_action(self, action: Action) -> Observation:
        obs = Observation(result="", active_tickets=[])
        try:
            handler = getattr(self, f"_handle_{action.action_type}", None)
            if handler: return handler(action, obs)
            else:
                obs.error = f"Unknown action: {action.action_type}"
                obs.result = "Invalid action."
                return obs
        except Exception as e:
            obs.error = str(e)
            obs.result = f"Error: {str(e)}"
            return obs

    def _handle_search_kb(self, action: Action, obs: Observation) -> Observation:
        if not action.query:
            obs.error = "query required"
            return obs
        results = search_knowledge_base(action.query)
        obs.kb_results = results
        obs.result = f"KB Results for '{action.query}':\n" + "\n\n".join(results)
        return obs

    def _handle_lookup_order(self, action: Action, obs: Observation) -> Observation:
        ticket = self._tickets.get(action.ticket_id)
        if not ticket: obs.error = "Invalid ticket"; return obs
        
        order_id = ticket.get("order_id")
        if order_id and order_id not in self._orders:
            order_data = lookup_order(order_id)
            if order_data: self._orders[order_id] = copy.deepcopy(order_data)
        
        if order_id in self._orders:
            obs.order_details = self._orders[order_id]
            obs.result = f"Order {order_id} Details: {self._orders[order_id]}"
        else:
            obs.result = "Order not found."
        
        if ticket["status"] == "open": ticket["status"] = "in_progress"
        return obs

    def _handle_reply(self, action: Action, obs: Observation) -> Observation:
        ticket = self._tickets.get(action.ticket_id)
        if not ticket or not action.message: obs.error = "Invalid ticket or message"; return obs
        
        ticket["interaction_history"].append({"role": "agent", "message": action.message})
        if ticket["status"] == "open": ticket["status"] = "in_progress"
        obs.result = f"Reply sent to ticket {action.ticket_id}."
        return obs

    def _handle_ask_info(self, action: Action, obs: Observation) -> Observation:
        ticket = self._tickets.get(action.ticket_id)
        if not ticket or not action.message: obs.error = "Invalid ticket/message"; return obs
        
        ticket["interaction_history"].append({"role": "agent", "message": f"[INFO REQUEST] {action.message}"})
        if ticket["status"] == "open": ticket["status"] = "in_progress"
        obs.result = f"Requested info on ticket {action.ticket_id}."
        return obs

    def _handle_refund(self, action: Action, obs: Observation) -> Observation:
        ticket = self._tickets.get(action.ticket_id)
        if not ticket: obs.error = "Invalid ticket"; return obs
        
        order = self._orders.get(ticket.get("order_id"))
        refund_amount = order["total"] if order else 0.0
        ticket["interaction_history"].append({"role": "system", "message": f"Refund of ${refund_amount:.2f} initiated."})
        if ticket["status"] == "open": ticket["status"] = "in_progress"
        obs.result = f"Refund initiated for ticket {action.ticket_id}."
        return obs

    def _handle_replace(self, action: Action, obs: Observation) -> Observation:
        ticket = self._tickets.get(action.ticket_id)
        if not ticket: obs.error = "Invalid ticket"; return obs
        
        ticket["interaction_history"].append({"role": "system", "message": "Replacement order created."})
        if ticket["status"] == "open": ticket["status"] = "in_progress"
        obs.result = f"Replacement created for ticket {action.ticket_id}."
        return obs

    def _handle_escalate(self, action: Action, obs: Observation) -> Observation:
        ticket = self._tickets.get(action.ticket_id)
        if not ticket: obs.error = "Invalid ticket"; return obs
        
        ticket["status"] = "escalated"
        ticket["interaction_history"].append({"role": "system", "message": f"Escalated. Reason: {action.reason}"})
        obs.result = f"Ticket {action.ticket_id} escalated."
        return obs

    def _handle_close(self, action: Action, obs: Observation) -> Observation:
        ticket = self._tickets.get(action.ticket_id)
        if not ticket: obs.error = "Invalid ticket"; return obs
        
        if ticket["status"] == "open":
            obs.result = f"Cannot close ticket {action.ticket_id} without any action."
            return obs
            
        ticket["status"] = "closed"
        ticket["interaction_history"].append({"role": "system", "message": "Ticket closed."})
        obs.result = f"Ticket {action.ticket_id} closed."
        return obs

    def _get_reward_reason(self, action: dict, reward: float) -> str:
        if reward > 0.15: return f"Good action: {action.get('action_type')}"
        elif reward > 0.05: return f"Partial progress: {action.get('action_type')}"
        elif reward > 0: return f"Small step: {action.get('action_type')}"
        elif reward < -0.1: return f"Penalty: {action.get('action_type')}"
        return f"Neutral: {action.get('action_type')}"
