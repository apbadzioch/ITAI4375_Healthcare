"""
fda_router.py  —  Tool-calling agent for the FDA Drug Label Assistant

Architecture:
    AgentChat  (the Gradio-facing class)
        └── FDAAgent  (orchestrates tool selection + execution)
                ├── tool: rag_query            — FAISS retrieval via ask()
                ├── tool: list_indexed_drugs   — what's in the index
                ├── tool: add_drug             — download + index on demand
                ├── tool: drug_comparison      — side-by-side multi-drug RAG
                └── tool: drug_section_lookup  — focused section retrieval

The agent uses Gemma3 via Ollama for tool-selection reasoning.
No LangChain agents are used here — this is a lightweight,
deterministic router with an LLM fallback for intent classification.
"""

import json
import re
from typing import Any

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Imports from your existing files — do NOT modify those files
from fda_part1 import (
    ask,
    indexed_drugs,
    add_drug_from_pdf,
    vector_store,
    build_filter,
    drug_prompt,
    llm,
)
from fda_ingest import ingest_drug

# ---------------------------------------------------------------------------
# TOOL DEFINITIONS
# Each tool has: name, description, and a callable handler.
# The router uses keyword heuristics first, LLM classification as fallback.
# ---------------------------------------------------------------------------

TOOLS = {
    "rag_query": {
        "description": (
            "Answer a question about one or more indexed drug labels. "
            "Use for: side effects, warnings, contraindications, dosage, "
            "interactions, pregnancy safety, overdose, pharmacology, indications."
        ),
    },
    "list_indexed_drugs": {
        "description": (
            "List all drug labels currently in the FAISS index. "
            "Use when the user asks what drugs are available, indexed, or loaded."
        ),
    },
    "add_drug": {
        "description": (
            "Download and index a new drug label PDF from DailyMed/openFDA. "
            "Use when the user wants to add a drug that isn't indexed yet."
        ),
    },
    "drug_comparison": {
        "description": (
            "Compare two or more drugs on a specific topic (e.g. side effects, "
            "dosage, interactions). Detected when the user names multiple drugs "
            "and asks a comparative question."
        ),
    },
    "drug_section_lookup": {
        "description": (
            "Pull a specific FDA label section for a drug: boxed warnings, "
            "indications, contraindications, overdosage, patient counseling, etc. "
            "Use when the user wants one precise section, not a general question."
        ),
    },
}


# ---------------------------------------------------------------------------
# TOOL HANDLERS
# ---------------------------------------------------------------------------

def tool_rag_query(query: str) -> str:
    """Delegate to the existing ask() function in fda_part1."""
    return ask(query)


def tool_list_indexed_drugs() -> str:
    if not indexed_drugs:
        return "No drug labels are currently indexed. Use 'Add Drug' to add one."
    drugs = sorted(indexed_drugs)
    bullet_list = "\n".join(f"  • {d}" for d in drugs)
    return f"**Indexed drug labels ({len(drugs)} total):**\n{bullet_list}"


def tool_add_drug(drug_name: str) -> str:
    """Download PDF via fda_ingest, then index via fda_part1."""
    name = drug_name.strip().title()

    if name in indexed_drugs:
        return f"**{name}** is already in the index. You can ask questions about it now."

    ok, result = ingest_drug(name)
    if not ok:
        return (
            f"❌ Could not download the label PDF for **{name}**.\n"
            f"Details: {result}\n\n"
            "Try checking the spelling or using the official generic/brand name."
        )

    status = add_drug_from_pdf(name, result)
    return status


def tool_drug_comparison(query: str, drugs: list[str]) -> str:
    """
    Run the same question against each drug independently,
    then ask the LLM to synthesise a comparison.
    """
    if not drugs:
        return "No drugs detected for comparison."

    results = {}
    for drug in drugs:
        targeted_query = f"{query} (for {drug})"
        results[drug] = ask(targeted_query)

    # Build a synthesis prompt
    blocks = "\n\n".join(
        f"=== {drug} ===\n{answer}" for drug, answer in results.items()
    )
    synthesis_prompt = (
        f"You are a clinical pharmacist. Below are FDA label excerpts for "
        f"{', '.join(drugs)} regarding: '{query}'.\n\n"
        f"{blocks}\n\n"
        f"Summarise the key similarities and differences in a clear, "
        f"structured comparison. Be factual and cite the drug names. "
        f"Flag any clinically significant differences."
    )
    comparison = llm.invoke(synthesis_prompt)
    return comparison


# Section keyword → section_id mapping (mirrors SECTION_MAP in fda_part1)
SECTION_KEYWORDS = {
    "boxed warning":       "BOXED_WARNING",
    "black box":           "BOXED_WARNING",
    "indication":          "INDICATIONS",
    "used for":            "INDICATIONS",
    "dosage":              "DOSAGE",
    "dose":                "DOSAGE",
    "administration":      "DOSAGE",
    "contraindication":    "CONTRAINDICATIONS",
    "warning":             "WARNINGS",
    "precaution":          "WARNINGS",
    "adverse":             "ADVERSE_REACTIONS",
    "side effect":         "ADVERSE_REACTIONS",
    "interaction":         "DRUG_INTERACTIONS",
    "pregnancy":           "SPECIFIC_POPS",
    "pediatric":           "SPECIFIC_POPS",
    "overdose":            "OVERDOSAGE",
    "overdosage":          "OVERDOSAGE",
    "pharmacology":        "PHARMACOLOGY",
    "mechanism":           "PHARMACOLOGY",
    "how supplied":        "HOW_SUPPLIED",
    "patient counseling":  "PATIENT_COUNSELING",
    "clinical studies":    "CLINICAL_STUDIES",
}


def tool_drug_section_lookup(drug_name: str, section_key: str, query: str) -> str:
    """
    Retrieve a specific FDA label section for a drug.
    Uses section-filtered FAISS retrieval + the standard drug_prompt.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_classic.chains.retrieval_qa.base import RetrievalQA
    from fda_part1 import vector_store as vs, embeddings

    if vs is None:
        return "No drugs are indexed yet."

    section_id = SECTION_KEYWORDS.get(section_key.lower(), None)
    if not section_id:
        # fall back to general ask
        return ask(query)

    meta_filter = {
        "drug":          drug_name,
        "section":       section_id,
        "is_short_chunk": False,
    }

    retriever = vs.as_retriever(
        search_kwargs={"k": 8, "filter": meta_filter}
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": drug_prompt},
    )
    result = qa_chain.invoke({"query": query})
    answer = result["result"]

    sources = result.get("source_documents", [])
    if sources:
        refs = set()
        for doc in sources:
            page = doc.metadata.get("page", "?")
            sec  = doc.metadata.get("section_name", section_key)
            refs.add(f"{drug_name} — {sec} (p.{page})")
        answer += "\n\n📄 Sources: " + " | ".join(sorted(refs))

    return answer


# ---------------------------------------------------------------------------
# INTENT ROUTER
# Heuristic-first, LLM-fallback classification.
# ---------------------------------------------------------------------------

# Compile a list of all indexed drug names for fast lookup
def _known_drugs_in_query(query: str) -> list[str]:
    q = query.lower()
    return [d for d in indexed_drugs if d.lower() in q]


# Patterns that clearly map to specific tools
_ADD_PATTERNS = re.compile(
    r"\b(add|index|ingest|download|load|get me|fetch)\b.*\b(drug|label|pdf|medication)\b"
    r"|\b(drug|label|pdf|medication)\b.*\b(add|index|ingest|download|load|fetch)\b"
    r"|add\s+\w+\s+to\s+(the\s+)?(index|system)",
    re.I,
)

_LIST_PATTERNS = re.compile(
    r"\b(list|show|what|which|display)\b.*(drug|label|indexed|available|loaded)"
    r"|(drug|label)s?\s+(available|indexed|loaded|in\s+the\s+index)",
    re.I,
)

_SECTION_PATTERNS = re.compile(
    r"\b(boxed warning|black box warning|indications?|contraindications?|"
    r"overdosage?|patient counseling|how supplied|clinical stud(?:y|ies)|"
    r"section\s+\d+)\b",
    re.I,
)

_COMPARE_PATTERNS = re.compile(
    r"\b(compar|versus|vs\.?|differ|between|both|either)\b",
    re.I,
)


# LLM-based fallback classifier (used when heuristics are ambiguous)
_ROUTER_PROMPT = PromptTemplate(
    input_variables=["tools", "query"],
    template="""You are a routing assistant for an FDA drug label Q&A system.
Given a user query, select the single best tool from this list:

{tools}

Query: "{query}"

Respond with ONLY the tool name (one of the keys listed above). No explanation.
Tool:""",
)


def classify_intent(query: str) -> dict[str, Any]:
    """
    Returns a dict:
        tool:    str               — which tool to call
        drug:    str | None        — primary drug if detected
        drugs:   list[str]         — all detected drugs (for comparison)
        section: str | None        — section keyword if relevant
        query:   str               — original query
    """
    detected = _known_drugs_in_query(query)

    # --- Hard heuristics first ---
    if _LIST_PATTERNS.search(query):
        return {"tool": "list_indexed_drugs", "drug": None, "drugs": [], "section": None, "query": query}

    if _ADD_PATTERNS.search(query):
        # Try to extract the drug name from the query
        # e.g. "add Metformin to the index"
        tokens = query.split()
        candidate = None
        for i, tok in enumerate(tokens):
            if tok.lower() in ("add", "index", "ingest", "download", "load", "fetch"):
                if i + 1 < len(tokens):
                    candidate = tokens[i + 1].strip(".,!?").title()
                    break
        return {"tool": "add_drug", "drug": candidate, "drugs": [], "section": None, "query": query}

    # Comparison: 2+ known drugs + comparative language
    if len(detected) >= 2 and _COMPARE_PATTERNS.search(query):
        return {"tool": "drug_comparison", "drug": detected[0], "drugs": detected, "section": None, "query": query}

    # Section lookup: one drug + explicit section mention
    if detected and _SECTION_PATTERNS.search(query):
        section_match = _SECTION_PATTERNS.search(query)
        section_key   = section_match.group(0).lower() if section_match else None
        return {"tool": "drug_section_lookup", "drug": detected[0], "drugs": detected, "section": section_key, "query": query}

    # General RAG: at least one known drug in query
    if detected:
        return {"tool": "rag_query", "drug": detected[0], "drugs": detected, "section": None, "query": query}

    # --- LLM fallback for ambiguous queries ---
    tools_str = "\n".join(f"  {k}: {v['description']}" for k, v in TOOLS.items())
    try:
        tool_name = llm.invoke(
            _ROUTER_PROMPT.format(tools=tools_str, query=query)
        ).strip().split()[0]  # take first word only
        if tool_name not in TOOLS:
            tool_name = "rag_query"
    except Exception:
        tool_name = "rag_query"

    return {"tool": tool_name, "drug": None, "drugs": detected, "section": None, "query": query}


# ---------------------------------------------------------------------------
# AGENT EXECUTOR
# ---------------------------------------------------------------------------

class FDAAgent:
    """
    Dispatches user queries to the right tool and returns a formatted response.
    """

    def run(self, query: str) -> str:
        intent = classify_intent(query)
        tool   = intent["tool"]

        # ── list_indexed_drugs ───────────────────────────────────────────
        if tool == "list_indexed_drugs":
            return tool_list_indexed_drugs()

        # ── add_drug ─────────────────────────────────────────────────────
        if tool == "add_drug":
            drug_name = intent.get("drug")
            if not drug_name:
                # Ask user to clarify
                return (
                    "Which drug would you like me to add to the index? "
                    "Please tell me the brand or generic name."
                )
            return tool_add_drug(drug_name)

        # ── drug_comparison ──────────────────────────────────────────────
        if tool == "drug_comparison":
            drugs = intent.get("drugs", [])
            if len(drugs) < 2:
                return (
                    "I detected a comparison request but only found one indexed drug in your query. "
                    "Please name at least two drugs that are already in the index, or add them first."
                )
            return tool_drug_comparison(query, drugs)

        # ── drug_section_lookup ──────────────────────────────────────────
        if tool == "drug_section_lookup":
            drug    = intent.get("drug")
            section = intent.get("section", "")
            if not drug:
                return ask(query)  # graceful fallback
            return tool_drug_section_lookup(drug, section or "", query)

        # ── rag_query (default) ──────────────────────────────────────────
        return tool_rag_query(query)


# ---------------------------------------------------------------------------
# GRADIO-FACING WRAPPER
# ---------------------------------------------------------------------------

class AgentChat:
    """
    Thin wrapper so fda_app.py can do:

        from fda_router import AgentChat
        agent = AgentChat()
        def respond(message, history):
            return agent.chat(message)
    """

    def __init__(self):
        self._agent = FDAAgent()

    def chat(self, message: str) -> str:
        if not message or not message.strip():
            return "Please enter a question or command."
        try:
            return self._agent.run(message.strip())
        except Exception as e:
            return (
                f"⚠️ An error occurred while processing your request.\n"
                f"Details: {e}\n\n"
                "Please try rephrasing your question."
            )
