import gradio as gr
from dataclasses import asdict
from environment import CustomerSupportEnv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()
api_env = CustomerSupportEnv(task_level="hard")

class StepRequest(BaseModel):
    action: str

# These API endpoints are required by openenv validation
@app.post("/reset")
def reset_env():
    result = api_env.reset()
    return {"observation": asdict(result.observation), "reward": result.reward, "done": result.done}

@app.post("/step")
def step_env(req: StepRequest):
    result = api_env.step(req.action)
    return {"observation": asdict(result.observation), "reward": result.reward, "done": result.done}

TICKET_TYPE_LABELS = {
    "order_status": "📦 Order Status Inquiry",
    "refund_request": "💰 Refund Request",
    "product_complaint": "😤 Product Complaint",
    "technical_issue": "🔧 Technical Issue",
    "billing_error": "💳 Billing Error"
}

def format_ticket_markdown(state):
    """Formats the state into a clean Markdown list with status indicators"""
    ticket_label = TICKET_TYPE_LABELS.get(state["ticket_type"], state["ticket_type"])
    sentiment = state["customer_sentiment"]
    severity = state["issue_severity"]
    wait_time = state["wait_time_hours"]
    
    # Emojis for quick visual feedback
    s_emoji = "🔴" if sentiment >= 8 else "🟡" if sentiment >= 5 else "🟢"
    sev_emoji = "🔴" if severity >= 8 else "🟡" if severity >= 5 else "🟢"
    w_emoji = "🔴" if wait_time >= 48 else "🟡" if wait_time >= 24 else "🟢"

    return f"""
### {ticket_label}
---
* **Customer Sentiment:** {s_emoji} `{sentiment} / 10`
* **Issue Severity:** {sev_emoji} `{severity} / 10`
* **Wait Time:** {w_emoji} `{wait_time} Hours`
    """

def create_app():
    # Use Gradio's native Soft theme with indigo/slate colors for a clean modern look
    theme = gr.themes.Soft(
        primary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
    )
    
    with gr.Blocks(theme=theme) as demo:
        gr.Markdown(
            """
            # 🛒 AI Customer Support Triage
            **Evaluate dynamic support tickets and choose the optimal action.**
            Balance urgency, customer sentiment, and issue severity to maximize your reward.
            """
        )
        
        env_state = gr.State(CustomerSupportEnv(task_level="hard"))
        
        with gr.Row():
            # Left Column (Live Feed)
            with gr.Column(scale=1):
                gr.Markdown("### 📡 Live Feed")
                with gr.Group():
                    ticket_display = gr.Markdown("Loading next ticket...")
                
            # Right Column (Command Center)
            with gr.Column(scale=1):
                gr.Markdown("### ⚡ Command Center")
                with gr.Group():
                    with gr.Row():
                        resolve_btn = gr.Button("✅ Resolve", variant="primary")
                        investigate_btn = gr.Button("🔍 Investigate", variant="secondary")
                    with gr.Row():
                        escalate_btn = gr.Button("🔺 Escalate", variant="stop")
                
                gr.Markdown("### 📈 Evaluation Metrics")
                with gr.Group():
                    status_output = gr.Textbox(label="Agent Status", interactive=False)
                    reward_output = gr.Textbox(label="Last Decision Score (0.00 to 1.00)", interactive=False)

        def load_state(env):
            state = env.state()
            return format_ticket_markdown(state), "Awaiting Agent Decision...", "N/A"
            
        def take_action(action, env):
            result = env.step(action)
            next_state = env.state()
            
            emoji = "✅" if action == "resolve" else "🔺" if action == "escalate" else "🔍"
            msg = f"{emoji} Ticket Action: {action.capitalize()}"
            reward_str = f"{result.reward:.2f}"
            
            return format_ticket_markdown(next_state), msg, reward_str, env

        # Initial load assignments
        demo.load(load_state, inputs=[env_state], outputs=[ticket_display, status_output, reward_output])
        
        # Action button assignments
        resolve_btn.click(lambda e: take_action("resolve", e), inputs=[env_state], outputs=[ticket_display, status_output, reward_output, env_state])
        escalate_btn.click(lambda e: take_action("escalate", e), inputs=[env_state], outputs=[ticket_display, status_output, reward_output, env_state])
        investigate_btn.click(lambda e: take_action("investigate", e), inputs=[env_state], outputs=[ticket_display, status_output, reward_output, env_state])
        
    return demo

gradio_app = create_app()

# Mount Gradio safely at the root. We bypass HF Spaces Gradio internal bugs using Docker SDK.
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
