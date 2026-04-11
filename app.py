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

custom_css = """
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
.gradio-container {
    max-width: 1000px !important;
}
.ticket-card {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
    transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.ticket-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 15px 50px -15px rgba(0, 0, 0, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.15);
}
.ticket-header {
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 20px;
    color: #38bdf8;
    text-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
    display: flex;
    align-items: center;
    gap: 10px;
}
.ticket-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    font-size: 1.15rem;
    padding: 10px 15px;
    border-radius: 8px;
    background: rgba(0,0,0,0.2);
    border: 1px solid rgba(255,255,255,0.02);
}
.stat-label {
    color: #94a3b8;
    font-weight: 500;
}
.stat-value {
    font-size: 1.25rem;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 6px;
}
.glass-panel {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.resolve-btn { background: linear-gradient(to right, #10b981, #059669) !important; color: white !important; border: none !important; }
.escalate-btn { background: linear-gradient(to right, #ef4444, #dc2626) !important; color: white !important; border: none !important; }
.investigate-btn { background: linear-gradient(to right, #3b82f6, #2563eb) !important; color: white !important; border: none !important; }
.action-btn {
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    transition: all 0.3s ease !important;
    opacity: 0.9 !important;
}
.action-btn:hover {
    transform: scale(1.03) !important;
    opacity: 1.0 !important;
    box-shadow: 0 0 20px rgba(255,255,255,0.15) !important;
}
.title-text {
    text-align: center;
    background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 900;
    margin-bottom: 5px;
}
.subtitle-text {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
    font-size: 1.1rem;
}
"""

def generate_html_card(state):
    ticket_label = TICKET_TYPE_LABELS.get(state["ticket_type"], state["ticket_type"])
    sentiment = state["customer_sentiment"]
    severity = state["issue_severity"]
    wait_time = state["wait_time_hours"]
    
    # Dynamic coloring relative to danger
    # Sentiment: 10 is furious (red), 1 is calm (green)
    s_bg = "rgba(239, 68, 68, 0.2)" if sentiment >= 8 else "rgba(234, 179, 8, 0.2)" if sentiment >= 5 else "rgba(34, 197, 94, 0.2)"
    s_col = "#fca5a5" if sentiment >= 8 else "#fde047" if sentiment >= 5 else "#86efac"
    
    # Severity: 10 is critical (red), 1 is trivial (green)
    sev_bg = "rgba(239, 68, 68, 0.2)" if severity >= 8 else "rgba(234, 179, 8, 0.2)" if severity >= 5 else "rgba(34, 197, 94, 0.2)"
    sev_col = "#fca5a5" if severity >= 8 else "#fde047" if severity >= 5 else "#86efac"
    
    # Wait time: >48 is red
    w_bg = "rgba(239, 68, 68, 0.2)" if wait_time >= 48 else "rgba(234, 179, 8, 0.2)" if wait_time >= 24 else "rgba(34, 197, 94, 0.2)"
    w_col = "#fca5a5" if wait_time >= 48 else "#fde047" if wait_time >= 24 else "#86efac"

    return f"""
    <div class="ticket-card">
        <div class="ticket-header">
            {ticket_label}
        </div>
        <div class="ticket-stat">
            <span class="stat-label">Customer Sentiment</span>
            <span class="stat-value" style="background: {s_bg}; color: {s_col};">{sentiment} / 10</span>
        </div>
        <div class="ticket-stat">
            <span class="stat-label">Issue Severity</span>
            <span class="stat-value" style="background: {sev_bg}; color: {sev_col};">{severity} / 10</span>
        </div>
        <div class="ticket-stat" style="margin-bottom: 0;">
            <span class="stat-label">Wait Time</span>
            <span class="stat-value" style="background: {w_bg}; color: {w_col};">{wait_time} Hours</span>
        </div>
    </div>
    """

def create_app():
    with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as demo:
        gr.HTML("<div class='title-text'>AI Customer Support Triage</div>")
        gr.HTML("<div class='subtitle-text'>Evaluate dynamic support tickets and choose the optimal action.</div>")
        
        env_state = gr.State(CustomerSupportEnv(task_level="hard"))
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📡 Live Feed")
                ticket_html = gr.HTML()
                
            with gr.Column(scale=1):
                gr.Markdown("### ⚡ Command Center")
                with gr.Group(elem_classes=["glass-panel"]):
                    with gr.Row():
                        resolve_btn = gr.Button("✅ Resolve", variant="primary", elem_classes=["action-btn", "resolve-btn"])
                        investigate_btn = gr.Button("🔍 Investigate", variant="secondary", elem_classes=["action-btn", "investigate-btn"])
                        escalate_btn = gr.Button("🔺 Escalate", variant="stop", elem_classes=["action-btn", "escalate-btn"])
                
                gr.Markdown("### 📈 Evaluation Metrics")
                with gr.Group(elem_classes=["glass-panel"]):
                    status_output = gr.Textbox(label="Agent Status", interactive=False)
                    reward_output = gr.Textbox(label="Last Decision Score (0.00 to 1.00)", interactive=False)

        def load_state(env):
            state = env.state()
            return generate_html_card(state), "Awaiting Agent Decision...", "N/A"
            
        def take_action(action, env):
            result = env.step(action)
            next_state = env.state()
            
            # Formulate response
            emoji = "✅" if action == "resolve" else "🔺" if action == "escalate" else "🔍"
            msg = f"{emoji} Ticket Action: {action.capitalize()}"
            reward_str = f"{result.reward:.2f}"
            
            return generate_html_card(next_state), msg, reward_str, env

        # Initial load bindings
        demo.load(load_state, inputs=[env_state], outputs=[ticket_html, status_output, reward_output])
        
        # Action button bindings
        resolve_btn.click(lambda e: take_action("resolve", e), inputs=[env_state], outputs=[ticket_html, status_output, reward_output, env_state])
        escalate_btn.click(lambda e: take_action("escalate", e), inputs=[env_state], outputs=[ticket_html, status_output, reward_output, env_state])
        investigate_btn.click(lambda e: take_action("investigate", e), inputs=[env_state], outputs=[ticket_html, status_output, reward_output, env_state])
        
    return demo

gradio_app = create_app()
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
