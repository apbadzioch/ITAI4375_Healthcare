"""
tools.py  —  LangChain @tool definitions for the FDA Drug Label Agent

All tools use the langchain_core @tool decorator, which handles:
  - Schema generation from docstrings + type hints
  - Integration with ChatOllama.bind_tools()
  - Automatic dispatch by LangGraph's ToolNode

Adding a new tool:
  1. Write a function with full type hints
  2. Decorate with @tool
  3. Add it to ALL_TOOLS at the bottom
  4. That's it — fda_agent.py picks up ALL_TOOLS automatically
"""

import json
import os

from langchain_core.tools import tool

from fda_part1 import ask, indexed_drugs, add_drug_from_pdf
from fda_ingest import ingest_drug


# ---------------------------------------------------------------------------
# TOOL: query_drug_label
# ---------------------------------------------------------------------------

@tool
def query_drug_label(question: str) -> str:
    """
    Search the indexed FDA drug label database to answer a clinical question.

    Use this for any question about a specific drug: side effects, dosage,
    warnings, contraindications, drug interactions, pregnancy safety,
    overdose, indications, mechanism of action, or pharmacology.

    Always try this tool first for any drug-related clinical question.

    Args:
        question: The clinical question to answer, e.g.
                  'What are the side effects of Metformin?'
                  'Is Lisinopril safe during pregnancy?'
                  'What are the boxed warnings for Warfarin?'
    """
    return ask(question)


# ---------------------------------------------------------------------------
# TOOL: list_indexed_drugs
# ---------------------------------------------------------------------------

@tool
def list_indexed_drugs() -> str:
    """
    Return the list of drug labels currently indexed and available to query.

    Use this when the user asks what drugs are available, what's in the
    database, or which drugs they can ask about.
    """
    if not indexed_drugs:
        return (
            "No drugs are currently indexed. "
            "Use add_drug to download and index a drug label."
        )
    drug_list = "\n".join(f"• {d}" for d in sorted(indexed_drugs))
    return f"Currently indexed drugs ({len(indexed_drugs)} total):\n{drug_list}"


# ---------------------------------------------------------------------------
# TOOL: add_drug
# ---------------------------------------------------------------------------

@tool
def add_drug(drug_name: str) -> str:
    """
    Download an FDA drug label PDF from DailyMed/openFDA and add it to the
    FAISS index so it can be queried.

    Use this when the user asks to add, index, or load a drug that is not
    already in the database. Takes 1-3 minutes depending on label length.

    Args:
        drug_name: Brand or generic name of the FDA-approved drug to add,
                   e.g. 'Metformin', 'Lipitor', 'Aspirin'.
    """
    name = drug_name.strip().title()

    if name in indexed_drugs:
        return f"'{name}' is already indexed and ready to query."

    ok, result = ingest_drug(name)
    if not ok:
        return f"Could not download the label for '{name}': {result}"

    status = add_drug_from_pdf(name, result)
    return status


# ---------------------------------------------------------------------------
# TOOL: get_drug_sections
# ---------------------------------------------------------------------------

@tool
def get_drug_sections(drug_name: str) -> str:
    """
    Return the FDA label sections available for an indexed drug, such as
    Warnings, Dosage and Administration, Adverse Reactions, Contraindications,
    Drug Interactions, and Clinical Pharmacology.

    Use this when the user wants to know what information is available for a
    specific drug, or wants to browse the label structure before asking
    detailed questions.

    Args:
        drug_name: Name of the drug to inspect. Must already be indexed,
                   e.g. 'Aspirin', 'Metformin'.
    """
    METADATA_DIR = "fda_metadata"
    name = drug_name.strip().title()

    # Find the metadata file, with a case-insensitive fallback
    path = os.path.join(METADATA_DIR, f"{name}_metadata.json")
    if not os.path.exists(path) and os.path.isdir(METADATA_DIR):
        for fname in os.listdir(METADATA_DIR):
            if fname.lower() == f"{name.lower()}_metadata.json":
                path = os.path.join(METADATA_DIR, fname)
                break

    if not os.path.exists(path):
        if name not in indexed_drugs:
            return f"'{name}' is not indexed. Use add_drug to index it first."
        return f"Metadata file not found for '{name}'."

    with open(path) as f:
        meta = json.load(f)

    sections = meta.get("sections", [])
    if not sections:
        return f"No section data found for '{name}'."

    lines = [
        f"Label sections for {name} "
        f"({meta.get('total_pages', '?')} pages, "
        f"{meta.get('total_chunks', '?')} chunks):"
    ]
    for sec in sections:
        lines.append(
            f"  - {sec['section_name']}  "
            f"(p.{sec['page_start']}-{sec['page_end']}, "
            f"{sec['chunk_count']} chunks)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# REGISTRY
# Import ALL_TOOLS in fda_agent.py — that's the only place that needs to change
# when you add a new tool.
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    query_drug_label,
    list_indexed_drugs,
    add_drug,
    get_drug_sections,
]
