"""
fda_app.py  —  FDA Drug Label Assistant (chat-only Gradio UI)

Single chat pane. The LangGraph agent (fda_agent.py) handles all logic via
tools defined in tools.py. Each browser session gets its own AgentSession
instance with isolated conversation history via LangGraph's MemorySaver.

To add a tool: see tools.py.
"""

import uuid
import gradio as gr

from fda_agent_1 import AgentSession

# ---------------------------------------------------------------------------
CSS = """
footer { display: none !important }

.custom-footer {
    text-align: center;
    padding: 12px;
    color: #7eb8f7;
    font-size: 0.82em;
    border-top: 1px solid #313244;
    margin-top: 8px;
}
"""

# ---------------------------------------------------------------------------
# SESSION FACTORY
# gr.State calls this once per browser session, giving each user an isolated
# AgentSession with its own thread_id for LangGraph's MemorySaver.

def make_session() -> AgentSession:
    return AgentSession(thread_id=str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# HANDLERS

def handle_submit(
    message: str,
    history: list[dict],
    session: AgentSession,
) -> tuple[list[dict], str]:
    """Send one user turn to the agent and append the reply to history."""
    if not message.strip():
        return history, ""
    reply = session.chat(message)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, ""


def handle_reset(session: AgentSession) -> tuple[list, AgentSession]:
    """Reset conversation history and clear the chatbot display."""
    session.reset()
    return [], session


# ---------------------------------------------------------------------------
# UI

with gr.Blocks(title="FDA Drug Label Assistant") as demo:

    session_state = gr.State(make_session)   # one AgentSession per browser tab

    gr.Markdown(
        "## 💊 FDA Drug Label Assistant\n"
        "Ask about side effects, dosage, warnings, contraindications, "
        "drug interactions, and more — sourced from FDA-approved labeling.\n\n"
        "You can also ask me to **add a new drug** "
        "(*'Add Lisinopril to the index'*) or **list available drugs**."
    )

    chatbot = gr.Chatbot(
        show_label=False,
        height=560,
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Ask about a drug, or say 'add Metformin'…",
            show_label=False,
            scale=8,
            container=False,
        )
        send_btn  = gr.Button("Send",     variant="primary",   scale=1)
        clear_btn = gr.Button("New Chat", variant="secondary", scale=1)

    gr.Examples(
        examples=[
            "What drugs are currently indexed?",
            "What are the side effects of Metformin?",
            "What are the contraindications for Lisinopril?",
            "What is the recommended dosage for Atorvastatin?",
            "Does Aspirin interact with blood thinners?",
            "Is Metformin safe during pregnancy?",
            "What are the boxed warnings for Warfarin?",
            "Show me the label sections for Aspirin",
        ],
        inputs=msg_box,
    )

    # Wire submit (Enter key and Send button share the same handler)
    msg_box.submit(
        handle_submit,
        inputs=[msg_box, chatbot, session_state],
        outputs=[chatbot, msg_box],
    )
    send_btn.click(
        handle_submit,
        inputs=[msg_box, chatbot, session_state],
        outputs=[chatbot, msg_box],
    )

    # New Chat resets the AgentSession (new thread_id) and clears the display
    clear_btn.click(
        handle_reset,
        inputs=[session_state],
        outputs=[chatbot, session_state],
    )

    gr.HTML(
        '<div class="custom-footer">'
        "💊 AI-Powered FDA Drug Label Analysis Engine.<br/>"
        "All information is sourced from publicly available FDA-approved labeling "
        "via openFDA &amp; DailyMed.<br/>"
        "<b>Not a substitute for professional medical advice.</b>"
        "</div>"
    )

demo.launch(share=True, css=CSS)
