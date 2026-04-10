import os
import json
from environment import CustomerSupportEnvironment
from models import Action

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def play_in_terminal():
    env = CustomerSupportEnvironment()
    obs = env.reset(task_id="hard")
    
    while True:
        clear_terminal()
        state = env.get_state()
        
        print("=== 🛒 AI Customer Support OpenEnv ===")
        print("           (Terminal Mode)          ")
        print("-" * 40)
        print("Observation:")
        print(obs.result)
        print("-" * 40)
        print("Active Tickets:")
        for t in obs.active_tickets:
            print(f"  - {t}")
        print("-" * 40)
        
        print("\nEnter your action JSON (or type 'exit' to quit):")
        print("Example: {\"action_type\": \"lookup_order\", \"ticket_id\": \"TKT-001\"}")
        
        action_str = input("> ").strip()
        
        if action_str.lower() == 'exit':
            print("\nExiting. Thank you for playing!")
            break
            
        try:
            action_dict = json.loads(action_str)
            action = Action(**action_dict)
            res = env.step(action)
            
            obs = res.observation
            reward = res.reward.value
            done = res.done
            
            print("\n" + "=" * 40)
            print(f"Action Processed!")
            print(f"Reward Received: {reward:.2f}")
            print(f"Reason: {res.reward.reason}")
            print("=" * 40)
            
            if done:
                print("\n🎉 Episode Complete!")
                print(obs.result)
                break
                
        except json.JSONDecodeError:
            print("\nInvalid JSON format!")
        except Exception as e:
            print(f"\nError processing action: {e}")
            
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    play_in_terminal()
