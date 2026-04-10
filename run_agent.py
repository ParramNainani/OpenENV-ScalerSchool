import random
import json
from environment import CustomerSupportEnvironment
from models import Action

def run_baseline_agent(num_episodes=5, task_level="medium"):
    env = CustomerSupportEnvironment()
    print(f"Running Baseline Agent for {num_episodes} episodes on '{task_level}' task...\n")
    
    total_reward = 0.0
    
    for episode in range(1, num_episodes + 1):
        obs = env.reset(task_id=task_level)
        done = False
        episode_reward = 0.0
        step_count = 0
        
        while not done and step_count < 10:
            active_tickets = obs.active_tickets if hasattr(obs, 'active_tickets') else []
            if not active_tickets:
                action_dict = {"action_type": "search_kb", "query": "help"}
            else:
                ticket_id = random.choice(active_tickets)
                action_type = random.choice(["lookup_order", "reply", "ask_info", "refund", "replace", "escalate", "close"])
                action_dict = {"action_type": action_type, "ticket_id": ticket_id, "message": "Automated response"}
            
            action = Action(**action_dict)
            res = env.step(action)
            
            episode_reward += res.reward.value
            obs = res.observation
            done = res.done
            step_count += 1
            
        total_reward += episode_reward
        print(f"Episode {episode}: Steps: {step_count} | Episode Reward: {episode_reward:.2f}")
            
    average_reward = total_reward / num_episodes
    print(f"\nAverage reward per episode: {average_reward:.2f}")

if __name__ == "__main__":
    random.seed(42)
    run_baseline_agent(num_episodes=5, task_level="medium")
