import os
from environment import CustomerSupportEnv

TICKET_TYPE_LABELS = {
    "order_status": "Order Status Inquiry",
    "refund_request": "Refund Request",
    "product_complaint": "Product Complaint",
    "technical_issue": "Technical Issue",
    "billing_error": "Billing Error"
}

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def play_in_terminal():
    # 'hard' mode includes all features (sentiment, severity, wait time, ticket type)
    env = CustomerSupportEnv(task_level="hard")
    
    while True:
        clear_terminal()
        state = env.state()
        print("=== AI Customer Support Triage ===")
        print("         (Terminal Mode)          ")
        print("-" * 40)
        print("Ticket Details:")
        print(f"  Type:             {TICKET_TYPE_LABELS.get(state['ticket_type'], state['ticket_type'])}")
        print(f"  Sentiment:        {state['customer_sentiment']}/10")
        print(f"  Severity:         {state['issue_severity']}/10")
        print(f"  Wait Time:        {state['wait_time_hours']} hours")
        print("-" * 40)
        
        print("\nAvailable Actions:")
        print("1. Resolve")
        print("2. Escalate")
        print("3. Investigate")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1/2/3/4): ").strip()
        
        if choice == '1':
            action = "resolve"
        elif choice == '2':
            action = "escalate"
        elif choice == '3':
            action = "investigate"
        elif choice == '4':
            print("\nExiting. Thank you for playing!")
            break
        else:
            print("\nInvalid choice. Press Enter to try again.")
            input()
            continue
            
        result = env.step(action)
        
        print("\n" + "=" * 40)
        print(f"Action Taken: {action.upper()}")
        print(f"Reward Received: {result.reward:.2f}")
        print("=" * 40)
        
        if result.reward == 1.0:
            print("Excellent decision! Maximum reward achieved.")
        elif result.reward == 0.0:
            print("Bad decision! This resulted in a poor outcome.")
        else:
            print("Okay decision. Partial reward received.")
            
        input("\nPress Enter to evaluate the next ticket...")

if __name__ == "__main__":
    play_in_terminal()
