import random
from environment import CustomerSupportEnv

def evaluate_task(task_level, num_episodes=100):
    """
    Evaluates a given task level with a dummy agent
    """
    env = CustomerSupportEnv(task_level=task_level)
    total_reward = 0.0
    
    for _ in range(num_episodes):
        state = env.state()
        action = random.choice(["resolve", "escalate", "investigate"])
        result = env.step(action)
        total_reward += result.reward
        
    return total_reward / num_episodes

def easy_task():
    print("Evaluating Task 1 - Easy...")
    reward = evaluate_task("easy")
    print(f"Task 1 Average Reward: {reward:.2f}\n")
    return reward

def medium_task():
    print("Evaluating Task 2 - Medium...")
    reward = evaluate_task("medium")
    print(f"Task 2 Average Reward: {reward:.2f}\n")
    return reward

def hard_task():
    print("Evaluating Task 3 - Hard...")
    reward = evaluate_task("hard")
    print(f"Task 3 Average Reward: {reward:.2f}\n")
    return reward

if __name__ == "__main__":
    # Ensure reproducible numbers
    random.seed(42)
    easy_task()
    medium_task()
    hard_task()
