"""
Tabs:
  1. Chat               — ask anything about indexed drug labels
  2. Indexed Drugs      — see what's in the index, refresh
  3. Add Drug           — enter a name, auto-download & index
  4. Charts             — side effects frequency chart per drug
"""

import gradio as gr
import plotly.graph_objects as go

from fda_part1 import (
    ask, indexed_drugs, extract_side_effects, add_drug_from_pdf
)
from fda_ingest import ingest_drug

# ---------------------------------------------------------------------------
# STYLING
css = """
footer {display: none !important}
.custom-footer {
    text-align: center;
    padding: 10px;
    color: #7eb8f7;
    font-size: 0.85em;
}
"""

chat = gr.Chatbot(show_label=False)

# ---------------------------------------------------------------------------
# CHAT FUNCTION
def respond(message, history):
    return ask(message)

# ---------------------------------------------------------------------------
# ADD DRUG FUNCTION
def handle_add_drug(drug_name: str):
    if not drug_name.strip():
        return "Please enter a drug name."

    name = drug_name.strip().title()

    # 1. Download PDF from openFDA / DailyMed
    ok, result = ingest_drug(name)
    if not ok:
        return f"❌ Download failed: {result}"

    pdf_path = result  # ingest_drug returns the path on success

    # 2. Index into FAISS
    status = add_drug_from_pdf(name, pdf_path)

    updated = "Indexed drugs: " + ", ".join(sorted(indexed_drugs))
    return f"{status}\n{updated}"


# ---------------------------------------------------------------------------
# SIDE-EFFECTS CHART FUNCTION
def generate_side_effects_chart(drug_name: str):
    if not drug_name:
        return None

    data = extract_side_effects(drug_name)

    if data is None or not data.get("adverse_reactions"):
        fig = go.Figure()
        fig.add_annotation(
            text=f"Could not extract side-effects for {drug_name}.<br>"
                 "Make sure the label is indexed and has an Adverse Reactions section.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        fig.update_layout(title=f"{drug_name} — Side Effects")
        return fig

    reactions = data["adverse_reactions"]
    # Sort by pct desc, take top 15
    reactions = sorted(reactions, key=lambda r: r.get("pct", 0), reverse=True)[:15]

    effects    = [r["effect"]           for r in reactions]
    pcts       = [r.get("pct", 0)       for r in reactions]
    freq_labels= [r.get("frequency", "") for r in reactions]

    # Color by rough frequency
    colors = []
    for r in reactions:
        f = r.get("frequency", "").lower()
        if "very common" in f or (r.get("pct", 0) or 0) >= 20:
            colors.append("#e05c5c")
        elif "common" in f or (r.get("pct", 0) or 0) >= 5:
            colors.append("#f0a04a")
        else:
            colors.append("#7eb8f7")

    hover = [
        f"<b>{e}</b><br>Frequency: {f}<br>Incidence: {p:.1f}%"
        for e, f, p in zip(effects, freq_labels, pcts)
    ]

    fig = go.Figure(go.Bar(
        x=pcts,
        y=effects,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
        text=[f"{p:.1f}%" if p else freq for p, freq in zip(pcts, freq_labels)],
        textposition="outside",
    ))

    fig.update_layout(
        title=f"{drug_name} — Adverse Reactions",
        xaxis_title="Incidence (%)",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=180, r=40, t=60, b=40),
        height=max(400, len(reactions) * 32 + 100),
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
    )
    return fig

# ---------------------------------------------------------------------------
# UI LAYOUT
with gr.Blocks(title="FDA Drug Label Assistant", css=css) as demo:

    with gr.Tabs():

        # ------------------------------------------------------------------
        # Tab 1: Chat
        with gr.Tab("Chat"):
            gr.ChatInterface(
                chatbot=chat,
                title="💊 FDA Drug Label Assistant",
                fn=respond,
                examples=[
                    "What are the side effects of Metformin?",
                    "What are the contraindications for Lisinopril?",
                    "What is the recommended dosage for Atorvastatin?",
                    "Does Aspirin interact with blood thinners?",
                    "Is Metformin safe during pregnancy?",
                    "What are the boxed warnings for this drug?",
                ],
            )

        # ------------------------------------------------------------------
        # Tab 2: Indexed Drugs
        with gr.Tab("Indexed Drugs"):
            gr.Markdown("### Drug labels currently in the index")
            indexed_display = gr.Textbox(
                value="\n".join(sorted(indexed_drugs)) or "No drugs indexed yet.",
                label="Indexed Drugs",
                interactive=False,
                lines=12,
            )
            refresh_btn = gr.Button("Refresh List")
            refresh_btn.click(
                fn=lambda: "\n".join(sorted(indexed_drugs)) or "No drugs indexed yet.",
                outputs=indexed_display,
            )

        # ------------------------------------------------------------------
        # Tab 3: Add a Drug
        with gr.Tab("Add Drug"):
            gr.Markdown("### Add a drug label to the index")
            gr.Markdown(
                "Enter the brand or generic name of any FDA-approved drug. "
                "The app will automatically download the label PDF from "
                "[DailyMed](https://dailymed.nlm.nih.gov) / "
                "[openFDA](https://open.fda.gov) and index it.\n\n"
                "> ⏳ Indexing takes 1–3 minutes depending on label length."
            )

            drug_input = gr.Textbox(
                label="Drug Name",
                placeholder="e.g. Metformin  or  Lipitor",
            )
            add_btn    = gr.Button("Download & Index", variant="primary")
            add_output = gr.Textbox(label="Status", interactive=False, lines=4)

            add_btn.click(
                fn=handle_add_drug,
                inputs=drug_input,
                outputs=add_output,
            )

        # ------------------------------------------------------------------
        # Tab 4: Charts
        with gr.Tab("Charts"):
            gr.Markdown("### Adverse Reactions — Frequency Breakdown")
            drug_selector = gr.Dropdown(
                choices=sorted(indexed_drugs),
                label="Select Drug",
                value=None,
            )
            effects_plot = gr.Plot(show_label=False)
            chart_btn    = gr.Button("Generate Chart", variant="primary")
            chart_btn.click(
                fn=generate_side_effects_chart,
                inputs=drug_selector,
                outputs=effects_plot,
            )

    gr.HTML(
        '<div class="custom-footer">'
        "💊 AI-Powered FDA Drug Label Analysis Engine.<br/>"
        "All information is sourced from publicly available FDA-approved labeling via openFDA &amp; DailyMed.<br/>"
        "<b>Not a substitute for professional medical advice.</b>"
        "</div>"
    )

demo.launch(share=True)
