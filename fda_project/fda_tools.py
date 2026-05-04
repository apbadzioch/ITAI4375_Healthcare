"""
fda_router.py  —  Tool-decorated agent for the FDA Drug Label Assistant

Uses LangChain's @tool decorator to define each capability as a
first-class tool, then a lightweight router selects and invokes them.

Tools:
    rag_query              — general drug label Q&A via FAISS
    list_indexed_drugs     — show what's in the index
    add_drug               — download + index a new drug label
    drug_comparison        — side-by-side multi-drug RAG + synthesis
    drug_section_lookup    — pinned retrieval for a specific FDA section
"""

import re
from typing import Optional

from langchain_core.tools import tool
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from fda_part1 import (
    ask,
    indexed_drugs,
    add_drug_from_pdf,
    vector_store,
    drug_prompt,
    llm,
)
from fda_ingest import ingest_drug


# ---------------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------------

@tool
def rag_query(query: str) -> str:
    """
    Answer any question about indexed FDA drug labels using RAG.
    Use for: side effects, warnings, contraindications, dosage,
    drug interactions, pregnancy safety, overdose, pharmacology,
    indications, and general drug information questions.
    """
    return ask(query)


@tool
def list_indexed_drugs() -> str:
    """
    Return a formatted list of all drug labels currently in the FAISS index.
    Use when the user asks what drugs are available, loaded, or indexed.
    """
    if not indexed_drugs:
        return "No drug labels are currently indexed. Use add_drug to add one."
    drugs = sorted(indexed_drugs)
    bullet_list = "\n".join(f"  • {d}" for d in drugs)
    return f"**Indexed drug labels ({len(drugs)} total):**\n{bullet_list}"


@tool
def add_drug(drug_name: str) -> str:
    """
    Download a drug label PDF from DailyMed/openFDA and index it into FAISS.
    Use when the user wants to add a drug that isn't in the index yet.
    The drug_name should be the brand or generic name, properly capitalised.
    """
    name = drug_name.strip().title()

    if name in indexed_drugs:
        return f"**{name}** is already indexed. You can ask questions about it now."

    ok, result = ingest_drug(name)
    if not ok:
        return (
            f"❌ Could not download the label PDF for **{name}**.\n"
            f"Details: {result}\n\n"
            "Try checking the spelling or using the official generic/brand name."
        )

    return add_drug_from_pdf(name, result)


@tool
def drug_comparison(query: str, drugs: list[str]) -> str:
    """
    Compare two or more indexed drugs on a specific topic.
    Use when the user names multiple drugs and asks a comparative question,
    e.g. 'compare the side effects of Metformin and Lisinopril'.
    drugs should be a list of drug names already present in the index.
    """
    if not drugs or len(drugs) < 2:
        return (
            "Comparison requires at least two indexed drugs. "
            "Please name two or more drugs that are already in the index, "
            "or add them first with add_drug."
        )

    results = {}
    for drug in drugs:
        results[drug] = ask(f"{query} (for {drug})")

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

    return llm.invoke(synthesis_prompt)


# Section keyword → FAISS metadata section_id
_SECTION_KEYWORDS: dict[str, str] = {
    "boxed warning":      "BOXED_WARNING",
    "black box":          "BOXED_WARNING",
    "indication":         "INDICATIONS",
    "used for":           "INDICATIONS",
    "dosage":             "DOSAGE",
    "dose":               "DOSAGE",
    "administration":     "DOSAGE",
    "contraindication":   "CONTRAINDICATIONS",
    "warning":            "WARNINGS",
    "precaution":         "WARNINGS",
    "adverse":            "ADVERSE_REACTIONS",
    "side effect":        "ADVERSE_REACTIONS",
    "interaction":        "DRUG_INTERACTIONS",
    "pregnancy":          "SPECIFIC_POPS",
    "pediatric":          "SPECIFIC_POPS",
    "overdose":           "OVERDOSAGE",
    "overdosage":         "OVERDOSAGE",
    "pharmacology":       "PHARMACOLOGY",
    "mechanism":          "PHARMACOLOGY",
    "how supplied":       "HOW_SUPPLIED",
    "patient counseling": "PATIENT_COUNSELING",
    "clinical studies":   "CLINICAL_STUDIES",
}


@tool
def drug_section_lookup(drug_name: str, section_key: str, query: str) -> str:
    """
    Retrieve a specific FDA label section for a single drug using
    section-pinned FAISS retrieval. More precise than rag_query.
    Use when the user explicitly names a section: boxed warning,
    contraindications, overdosage, patient counseling, clinical studies, etc.
    drug_name: the indexed drug name.
    section_key: the section keyword (e.g. 'boxed warning', 'overdosage').
    query: the original user question.
    """
    if vector_store is None:
        return "No drugs are indexed yet."

    section_id = next(
        (v for k, v in _SECTION_KEYWORDS.items() if k in section_key.lower()),
        None,
    )

    if not section_id:
        return ask(query)  # graceful fallback to general RAG

    meta_filter = {
        "drug":           drug_name,
        "section":        section_id,
        "is_short_chunk": False,
    }

    retriever = vector_store.as_retriever(
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
        refs = {
            f"{drug_name} — {doc.metadata.get('section_name', section_key)} "
            f"(p.{doc.metadata.get('page', '?')})"
            for doc in sources
        }
        answer += "\n\n📄 Sources: " + " | ".join(sorted(refs))

    return answer


# ---------------------------------------------------------------------------
# TOOL REGISTRY
# Makes it easy to iterate, inspect, or bind tools to an LLM later.
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    rag_query,
    list_indexed_drugs,
    add_drug,
    drug_comparison,
    drug_section_lookup,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}


# ---------------------------------------------------------------------------
# ROUTER
# Heuristic-first, LLM-fallback intent classification.
# ---------------------------------------------------------------------------

_ADD_RE = re.compile(
    r"\b(add|index|ingest|download|load|fetch)\b.{0,30}\b(drug|label|medication)\b"
    r"|\badd\s+\w+\s+to\s+(the\s+)?(index|system)\b",
    re.I,
)
_LIST_RE = re.compile(
    r"\b(list|show|what|which|display)\b.{0,20}(drug|label|indexed|available|loaded)"
    r"|(drug|label)s?\s+(available|indexed|loaded|in\s+the\s+index)",
    re.I,
)
_SECTION_RE = re.compile(
    r"\b(boxed warning|black box warning|indications?|contraindications?|"
    r"overdosage?|patient counseling|how supplied|clinical stud(?:y|ies)|"
    r"section\s+\d+)\b",
    re.I,
)
_COMPARE_RE = re.compile(
    r"\b(compar|versus|vs\.?|differ|between|both|either)\b",
    re.I,
)

# One-liner descriptions for the LLM fallback prompt
_ROUTER_SYSTEM = (
    "You are a routing assistant for an FDA drug label Q&A system. "
    "Given a user query, respond with ONLY the name of the best tool "
    "from this list — no explanation, no punctuation, just the tool name:\n\n"
    + "\n".join(
        f"  {t.name}: {t.description.splitlines()[0]}"
        for t in ALL_TOOLS
    )
)


def _known_drugs(query: str) -> list[str]:
    q = query.lower()
    return [d for d in indexed_drugs if d.lower() in q]


def _extract_drug_from_add_query(query: str) -> Optional[str]:
    """Best-effort extraction of a drug name from an add/index request."""
    triggers = {"add", "index", "ingest", "download", "load", "fetch"}
    tokens = query.split()
    for i, tok in enumerate(tokens):
        if tok.lower() in triggers and i + 1 < len(tokens):
            return tokens[i + 1].strip(".,!?").title()
    return None


def route(query: str) -> dict:
    """
    Classify user intent and return:
        {
            "tool":   str,   — key into TOOL_MAP (or "_clarify")
            "kwargs": dict,  — ready to pass to tool.invoke()
        }
    """
    detected = _known_drugs(query)

    if _LIST_RE.search(query):
        return {"tool": "list_indexed_drugs", "kwargs": {}}

    if _ADD_RE.search(query):
        drug = _extract_drug_from_add_query(query)
        if not drug:
            return {
                "tool":   "_clarify",
                "kwargs": {"message": "Which drug would you like me to add? Please give me the brand or generic name."},
            }
        return {"tool": "add_drug", "kwargs": {"drug_name": drug}}

    if len(detected) >= 2 and _COMPARE_RE.search(query):
        return {"tool": "drug_comparison", "kwargs": {"query": query, "drugs": detected}}

    if detected and _SECTION_RE.search(query):
        section_match = _SECTION_RE.search(query)
        return {
            "tool": "drug_section_lookup",
            "kwargs": {
                "drug_name":   detected[0],
                "section_key": section_match.group(0).lower(),
                "query":       query,
            },
        }

    if detected:
        return {"tool": "rag_query", "kwargs": {"query": query}}

    # LLM fallback for ambiguous queries with no known drug name
    try:
        tool_name = llm.invoke(f"{_ROUTER_SYSTEM}\n\nQuery: {query}").strip().split()[0]
        if tool_name not in TOOL_MAP:
            tool_name = "rag_query"
    except Exception:
        tool_name = "rag_query"

    return {"tool": tool_name, "kwargs": {"query": query}}


# ---------------------------------------------------------------------------
# AGENT EXECUTOR
# ---------------------------------------------------------------------------

class FDAAgent:
    def run(self, query: str) -> str:
        decision  = route(query)
        tool_name = decision["tool"]
        kwargs    = decision["kwargs"]

        if tool_name == "_clarify":
            return kwargs["message"]

        tool_fn = TOOL_MAP.get(tool_name)
        if not tool_fn:
            return ask(query)  # safe fallback

        return tool_fn.invoke(kwargs)


# ---------------------------------------------------------------------------
# GRADIO-FACING WRAPPER  (matches the import in fda_app.py unchanged)
# ---------------------------------------------------------------------------

class AgentChat:
    """
    Drop-in replacement. fda_app.py imports this unchanged:

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
                f"⚠️ Something went wrong processing your request.\n"
                f"Details: {e}\n\n"
                "Try rephrasing your question."
            )
