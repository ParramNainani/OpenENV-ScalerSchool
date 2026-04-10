import json
import gradio as gr
from fastapi import FastAPI
import uvicorn

from models import Action
from environment import CustomerSupportEnvironment

app = FastAPI()
env = CustomerSupportEnvironment()

@app.get("/state")
def get_state():
    return env.get_state().model_dump() if hasattr(env.get_state(), 'model_dump') else env.get_state().dict()

@app.post("/reset")
def reset(task_id: str = "easy"):
    return env.reset(task_id).model_dump() if hasattr(env.reset(task_id), 'model_dump') else env.reset(task_id).dict()

@app.post("/step")
def step(action: Action):
    return env.step(action).model_dump() if hasattr(env.step(action), 'model_dump') else env.step(action).dict()

def gradio_reset(task_id):
    obs = env.reset(task_id)
    return obs.result, str(env.get_state().tickets), "0.0"

def gradio_step(action_str):
    try:
        action_dict = json.loads(action_str)
        action = Action(**action_dict)
        res = env.step(action)
        return res.observation.result, str(env.get_state().tickets), str(res.reward.value)
    except Exception as e:
        return f"Error: {e}", str(env.get_state().tickets), "0.0"

ACTION_TEMPLATES = {
    "Look up Order Details": '{"action_type": "lookup_order", "ticket_id": "TKT-001"}',
    "Search Knowledge Base": '{"action_type": "search_kb", "query": "refund policy"}',
    "Reply to Customer": '{"action_type": "reply", "ticket_id": "TKT-001", "message": "Hi! I checked your order and it should arrive soon."}',
    "Ask for Information": '{"action_type": "ask_info", "ticket_id": "TKT-001", "message": "Could you provide a photo of the damaged item?"}',
    "Issue Refund": '{"action_type": "refund", "ticket_id": "TKT-001"}',
    "Issue Replacement": '{"action_type": "replace", "ticket_id": "TKT-001"}',
    "Escalate to Supervisor": '{"action_type": "escalate", "ticket_id": "TKT-001", "reason": "Customer is extremely angry"}',
    "Close Ticket": '{"action_type": "close", "ticket_id": "TKT-001"}'
}

def create_app():
    with gr.Blocks(title="AI Customer Support OpenEnv") as demo:
        gr.Markdown("# 🛒 AI Customer Support OpenEnv Dashboard")
        with gr.Row():
            with gr.Column():
                task_dropdown = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy", label="Task Difficulty")
                reset_btn = gr.Button("Reset Environment")
                
                template_dropdown = gr.Dropdown(
                    choices=list(ACTION_TEMPLATES.keys()),
                    label="Action Template Shortcuts (Pick one to auto-fill JSON)",
                    value="Look up Order Details"
                )
                
                action_input = gr.Textbox(
                    label="Action JSON",
                    value=ACTION_TEMPLATES["Look up Order Details"],
                    lines=3
                )
                step_btn = gr.Button("Execute Action")
                
            with gr.Column():
                output_obs = gr.Textbox(label="Observation Result", lines=5)
                output_reward = gr.Textbox(label="Step Reward")
                output_state = gr.Textbox(label="Active Tickets State", lines=10)

        reset_btn.click(gradio_reset, inputs=[task_dropdown], outputs=[output_obs, output_state, output_reward])
        step_btn.click(gradio_step, inputs=[action_input], outputs=[output_obs, output_state, output_reward])
        template_dropdown.change(fn=lambda k: ACTION_TEMPLATES.get(k, ""), inputs=[template_dropdown], outputs=[action_input])
    
    return demo

gradio_app = create_app()
gradio_app.queue()
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
