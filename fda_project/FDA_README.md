# FDA Drug Label RAG System

An AI-powered drug label analysis engine built on the same architecture as
the SEC/10-K chatbot — swapping financial filings for FDA-approved drug labels.

---

## Files

| File | Purpose | Mirrors |
|------|---------|---------|
| `fda_part1.py` | RAG core — embeddings, FAISS, LLM, prompts, ask() | `part1.py` |
| `fda_ingest.py` | Download label PDFs from openFDA + DailyMed | *(new)* |
| `fda_app.py` | Gradio UI — Chat, Indexed Drugs, Add Drug, Charts | `app.py` |

---

## Quick Start

### 1. Install dependencies

```bash
pip install gradio langchain langchain-community langchain-huggingface \
            langchain-ollama faiss-cpu sentence-transformers \
            requests plotly pypdf
```

> Same stack as the SEC chatbot. You already have most of this.

### 2. Pre-download some drug labels (optional but faster)

```bash
python fda_ingest.py "Metformin" "Lisinopril" "Atorvastatin" "Aspirin"
```

PDFs land in `fda_pdfs/`. They're auto-indexed on first app launch.

### 3. Launch the app

```bash
python fda_app.py
```

Then open the local URL. The **Add Drug** tab lets you download + index
any drug by name without touching the terminal.

---

## Architecture

```
User query
    │
    ▼
detect_drug()          ← scans query for known drug names
    │
    ▼
build_filter()         ← routes to the right FDA label section
    │                    (ADVERSE_REACTIONS, WARNINGS, DOSAGE, etc.)
    ▼
FAISS retriever        ← k=6 semantically similar chunks
    │
    ▼
RetrievalQA chain      ← gemma3:27b-cloud + drug_prompt
    │
    ▼
answer + sources       ← section name + page number citations
```

### Section-aware routing (mirrors 10-K Item routing)

| Query keywords | Routed to |
|---------------|-----------|
| "side effect", "adverse", "reaction" | ADVERSE_REACTIONS |
| "warning", "black box" | WARNINGS |
| "contraindic", "avoid", "should not" | CONTRAINDICATIONS |
| "dose", "dosage", "how much" | DOSAGE |
| "interact", "taken with" | DRUG_INTERACTIONS |
| "pregnant", "pregnancy" | SPECIFIC_POPS (mentions_pregnancy filter) |
| "child", "pediatric" | SPECIFIC_POPS (mentions_pediatric filter) |
| "overdose" | OVERDOSAGE |
| "what is", "used for", "indication" | INDICATIONS |
| "mechanism", "how does it work" | PHARMACOLOGY |

---

## openFDA + DailyMed

- **openFDA** (`api.fda.gov/drug/label.json`) — search index, finds `set_id`
- **DailyMed** (`dailymed.nlm.nih.gov`) — canonical source for label PDFs

Both are free, public APIs. No key required.

---

## Notes / Tradeoffs

- **Label quality varies** — older drugs may have scanned (non-searchable) PDFs.
  If a drug returns poor answers, check the PDF manually in `fda_pdfs/`.

- **LLM is the same Gemma3:27b-cloud** you're already running. Temperature
  is set to 0.1 (vs 0.2 for finance) for more conservative clinical answers.

- **Not medical advice.** The footer says it. The system cites label sections
  so users can verify against the source document.
