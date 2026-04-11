def calculate_reward(state, action, task_level="easy"):
    """
    Calculate reward between 0.0 and 1.0 based on state, action, and task difficulty.
    Actions: "resolve", "escalate", "investigate"
    """
    ticket_type = state["ticket_type"]
    sentiment = state["customer_sentiment"]
    severity = state["issue_severity"]
    wait_time = state["wait_time_hours"]
    
    if action not in ["resolve", "escalate", "investigate"]:
        return 0.0
        
    if task_level == "easy":
        # Task 1 - Easy: simple rule-based grading on severity
        if severity <= 3:
            correct_action = "resolve"
        elif severity >= 8:
            correct_action = "escalate"
        else:
            correct_action = "investigate"
            
        if action == correct_action:
            return 1.0
        else:
            return 0.0
            
    elif task_level == "medium":
        # Task 2 - Medium: Multiple features (sentiment + severity)
        urgency = (sentiment + severity) / 2
        
        is_simple = severity <= 3 and sentiment <= 4
        is_critical = severity >= 8 or (sentiment >= 8 and severity >= 5)
        
        if action == "resolve":
            if is_simple:
                return 0.9  # Safe resolution
            elif is_critical:
                return 0.0  # Dangerous to resolve critical issues directly
            else:
                return 0.5  # Moderate risk resolution
        elif action == "escalate":
            if is_critical:
                return 1.0  # Correct escalation
            elif is_simple:
                return 0.0  # Wasted escalation
            else:
                return 0.3  # Over-cautious
        elif action == "investigate":
            if not is_simple and not is_critical:
                return 1.0  # Correct investigation
            else:
                return 0.3  # Partial progress
                
    elif task_level == "hard":
        # Task 3 - Hard: VIP risk and complex triage
        is_vip_risk = wait_time > 48 and sentiment >= 7 and severity >= 6
        
        is_critical = severity >= 8 or (sentiment >= 8 and severity >= 5)
        is_simple = severity <= 3 and sentiment <= 4
        
        if is_vip_risk:
            # Long-waiting angry customer — must escalate immediately
            if action == "escalate":
                return 1.0  # VIP risk handled
            elif action == "investigate":
                return 0.5  # Partial, noticed something off
            else:
                return 0.0  # Resolving without care (bad)
                
        if action == "resolve":
            if is_simple:
                return 1.0
            elif is_critical:
                return 0.0  # Dangerous resolution
            else:
                return 0.2  # Risky resolution
        elif action == "escalate":
            if is_critical:
                return 1.0
            elif is_simple:
                return 0.0
            else:
                return 0.3  # Partial progress
        elif action == "investigate":
            if not is_simple and not is_critical and not is_vip_risk:
                return 1.0
            elif is_simple:
                return 0.0
            else:
                return 0.5  # Partial progress
    
    return 0.0
