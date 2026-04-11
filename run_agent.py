import random
from environment import CustomerSupportEnv

def run_baseline_agent(num_episodes=50, task_level="medium"):
    env = CustomerSupportEnv(task_level=task_level)
    print(f"Running Baseline Agent for {num_episodes} episodes on '{task_level}' task...\n")
    
    total_reward = 0.0
    
    for episode in range(1, num_episodes + 1):
        state = env.state()
        # Random Agent Baseline
        action = random.choice(["resolve", "escalate", "investigate"])
        result = env.step(action)
        
        total_reward += result.reward
        
        if episode <= 10:  # Print only first 10 steps to keep output concise
            print(f"Episode {episode}: State: {state} | Action: {action} | Reward: {result.reward:.2f}")
            
    if num_episodes > 10:
        print("...")
        
    average_reward = total_reward / num_episodes
    print(f"\nAverage reward: {average_reward:.2f}")

if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    run_baseline_agent(num_episodes=50, task_level="medium")
