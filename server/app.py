import gradio as gr
from fastapi import FastAPI
import uvicorn
import json

from models import Action
from environment import CustomerSupportEnvironment

env = CustomerSupportEnvironment()
app = FastAPI()

# ---- FastAPI Endpoints ----
@app.get("/state")
def get_state(): return env.get_state().model_dump() if hasattr(env.get_state(), 'model_dump') else env.get_state().dict()

@app.post("/reset")
def reset(task_id: str = "easy"): return env.reset(task_id).model_dump() if hasattr(env.reset(task_id), 'model_dump') else env.reset(task_id).dict()

@app.post("/step")
def step(action: Action): return env.step(action).model_dump() if hasattr(env.step(action), 'model_dump') else env.step(action).dict()

# ---- Gradio UI ----
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

with gr.Blocks(title="AI Customer Support OpenEnv") as demo:
    gr.Markdown("# 🛒 AI Customer Support OpenEnv Dashboard")
    with gr.Row():
        with gr.Column():
            task_dropdown = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy", label="Task Difficulty")
            reset_btn = gr.Button("Reset Environment")
            
            action_input = gr.Textbox(
                label="Action JSON",
                value='{"action_type": "lookup_order", "ticket_id": "TKT-001"}',
                lines=3
            )
            step_btn = gr.Button("Execute Action")
            
        with gr.Column():
            output_obs = gr.Textbox(label="Observation Result", lines=5)
            output_reward = gr.Textbox(label="Step Reward")
            output_state = gr.Textbox(label="Active Tickets State", lines=10)

    reset_btn.click(gradio_reset, inputs=[task_dropdown], outputs=[output_obs, output_state, output_reward])
    step_btn.click(gradio_step, inputs=[action_input], outputs=[output_obs, output_state, output_reward])

# Mount Gradio to FastAPI
demo.queue()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
