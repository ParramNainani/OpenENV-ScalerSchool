def calculate_reward(state, action, task_level="easy"):
    """
    Calculate a continuous smooth reward between 0.0 and 1.0 based on state, action, and task difficulty.
    Actions: "resolve", "escalate", "investigate"
    """
    ticket_type = state["ticket_type"]
    sentiment = state["customer_sentiment"]
    severity = state["issue_severity"]
    wait_time = state["wait_time_hours"]
    
    if action not in ["resolve", "escalate", "investigate"]:
        return 0.0
        
    if task_level == "easy":
        # Task 1 - Easy: Continuous scoring on severity
        s_norm = (severity - 1) / 9.0  # Range: [0.0, 1.0]
        
        if action == "resolve":
            reward = 1.0 - (s_norm * 1.5)
        elif action == "escalate":
            reward = (s_norm * 1.2) - 0.1
        elif action == "investigate":
            # Parabolic scoring favoring middle severity
            reward = 1.0 - (abs(s_norm - 0.5) * 2.0)
            
        return max(0.0, min(1.0, float(reward)))
            
    elif task_level == "medium":
        # Task 2 - Medium: Blended urgency (sentiment + severity)
        urgency = ((severity - 1) / 9.0 + (sentiment - 1) / 9.0) / 2.0  # Range: [0.0, 1.0]
        
        if action == "resolve":
            # Highly punished rapidly if urgency starts rising
            reward = 1.0 - (urgency * 2.0)
        elif action == "escalate":
            # Smoothly increases towards 1.0 as urgency maxes
            reward = (urgency * 1.8) - 0.2
        elif action == "investigate":
            # Bell-curve favoring moderate ambiguity (around 0.4 urgency)
            reward = 1.0 - (abs(urgency - 0.4) * 2.2)
            
        return max(0.0, min(1.0, float(reward)))
            
    elif task_level == "hard":
        # Task 3 - Hard: VIP risk and complex dynamic triage (wait time added)
        w_score = min(wait_time / 48.0, 1.0) 
        s_score = (severity - 1) / 9.0
        sent_score = (sentiment - 1) / 9.0
        
        # Weighted comprehensive urgency factor
        urgency = (s_score * 0.45 + sent_score * 0.35 + w_score * 0.20)
        
        if action == "resolve":
            reward = 1.0 - (urgency * 2.5)
        elif action == "escalate":
            reward = (urgency * 1.8) - 0.1
            
            # Massive bonus for catching VIP risks (Waiting over 48h and highly irritated)
            if w_score > 0.8 and sent_score > 0.8:
                reward += 0.35
                
        elif action == "investigate":
            # Highest reward on middle-band urgency where triage is needed
            reward = 1.0 - (abs(urgency - 0.45) * 2.5)
            
            # Heavy penalty if investigating instead of escalating a severe case
            if urgency > 0.8:
                reward -= 0.5
                
        return max(0.0, min(1.0, float(reward)))
    
    return 0.0
