"""
an idea upgrading to agent style using langgraph
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_ollama import OllamaLLM
from langchain import PromptTemplate

# keep existing embeddings, vector_store, and SECTION MAP
"""
TOOLS: wrap logic into @tool decorators
the docstrings are CRITICAL; this is how the agent "knows" which tool to use
"""
@tool
def search_drug_label(drug_name: str, section_id: str = None, query: str = None):
    """
    Search for specific info within a drug's FDA label.
    Provide the drug_name and an optional section_id (e.g. 'WARNINGS', 'DOSAGE', 'ADVERSE_REACTIONS').
    if section_id is omitted, perform a general search across the whole label.
    """
    if vector_store is None:
        return "Error: Not Initialized."
    # the agent now decides the filter, not a hardcoded Python if/else block
    meta_filter = {"drug": drug_name}
    if section_id:
        meta_filter["section_id"] = section_id

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5, "filter": meta_filter}
    )

    # use simple retriever instead of full QA chain to give agent raw context
    docs = retriever.get_relevant_documents(query or drug_name)
    context = "\n\n".join([doc.page_content for doc in docs])

    if not context:
        return "No relevant info found in the section."

    return context

@tool
def get_drug_sections(drug_name: str):
    """
    returns a list of available sections for a specific drug to help the agent narrow down searches.
    """
    # use existing METADATA_DIR logic
    path = os.path.join(METADATA_DIR, f"{drug_name}_metadata.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            sections = [s["section_id"] for s in data["sections"]]
            return f"Available sections for {drug_name}: {', '.join(sections)}"
    return f"Drug {drug_name} not found in index."

@tool
def extract_side_effects_data(drug_name: str):
    """
    Extracts a structured list of side effects and percentages for a specific drug.
    """
    # call existing extract_side_effects(drug_name)
    return extract_side_effects(drug_name)

@tool
def index_new_drug(drug_name: str, pdf_path: str):
    """
    Indexes a new PDF into the system, use this if a drug is missing from the registry.
    """
    return add_drug_from_pdf(drug_name, pdf_path)


# -----------------------------------------------------------
# AGENT SETUP
# -----------------------------------------------------------

