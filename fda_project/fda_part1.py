"""
fda_part1.py  —  RAG core for openFDA drug-label PDFs
Mirrors the architecture of part1.py (SEC/10-K) but adapted for
pharmaceutical / drug-safety domain.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

import os, json, re
from datetime import datetime

# ---------------------------------------------------------------------------
# PATHS
INDEX_PATH   = "fda_faiss_index"
METADATA_DIR = "fda_metadata"
REGISTRY     = "fda_indexed_drugs.json"
PDF_DIR      = "fda_pdfs"            # drug label PDFs land here after download

# ---------------------------------------------------------------------------
# SPLITTER  (drug labels are dense — tighter chunks help precision)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120,
)

# ---------------------------------------------------------------------------
# EMBEDDINGS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------------------
# LLM  (same cloud Gemma3 you're already using)
llm = OllamaLLM(
    model="gemma4:31b-cloud",
    temperature=0.1,           # lower temp → more precise for clinical info
)

# ---------------------------------------------------------------------------
# PROMPTS

drug_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a clinical pharmacist assistant specialized in FDA-approved drug labeling.
Use ONLY the provided drug label context to answer.
If the answer is not in the context, say so clearly — do NOT guess or hallucinate.
Always cite the specific section (e.g. Warnings, Adverse Reactions, Dosage) when available.
Be precise and factual. This is medical reference information.

Context:
{context}

Question:
{question}

Answer:
""",
)

# Used for the side effects bar chart extraction
side_effects_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
{question}
Drug label context: {context}
Respond ONLY with a JSON object. No explanation, no markdown fences.
""",
)

# ---------------------------------------------------------------------------
# SECTION MAP  — for FDA label structure
# Standard FDA drug label sections (21 CFR 201.56 / 201.57)
SECTION_MAP = [
    (r"boxed\s*warning|black\s*box",         "BOXED_WARNING",      "Boxed Warning"),
    (r"^1\s+indications?\s+and\s+usage",     "INDICATIONS",        "Indications and Usage"),
    (r"^2\s+dosage\s+and\s+admin",           "DOSAGE",             "Dosage and Administration"),
    (r"^3\s+dosage\s+forms?\s+and\s+strength","DOSAGE_FORMS",      "Dosage Forms and Strengths"),
    (r"^4\s+contraindications?",             "CONTRAINDICATIONS",  "Contraindications"),
    (r"^5\s+warnings?\s+and\s+precautions?", "WARNINGS",           "Warnings and Precautions"),
    (r"^6\s+adverse\s+reactions?",           "ADVERSE_REACTIONS",  "Adverse Reactions"),
    (r"^7\s+drug\s+interactions?",           "DRUG_INTERACTIONS",  "Drug Interactions"),
    (r"^8\s+use\s+in\s+specific",            "SPECIFIC_POPS",      "Use in Specific Populations"),
    (r"^9\s+drug\s+abuse",                   "ABUSE",              "Drug Abuse and Dependence"),
    (r"^10\s+overdosage",                    "OVERDOSAGE",         "Overdosage"),
    (r"^11\s+description",                   "DESCRIPTION",        "Description"),
    (r"^12\s+clinical\s+pharmacology",       "PHARMACOLOGY",       "Clinical Pharmacology"),
    (r"^13\s+nonclinical\s+toxicology",      "TOXICOLOGY",         "Nonclinical Toxicology"),
    (r"^14\s+clinical\s+studies",            "CLINICAL_STUDIES",   "Clinical Studies"),
    (r"^16\s+how\s+supplied",                "HOW_SUPPLIED",       "How Supplied"),
    (r"^17\s+patient\s+counseling",          "PATIENT_COUNSELING", "Patient Counseling"),
    # fallback patterns (not anchored)
    (r"indications?\s+and\s+usage",          "INDICATIONS",        "Indications and Usage"),
    (r"adverse\s+reactions?",                "ADVERSE_REACTIONS",  "Adverse Reactions"),
    (r"warnings?\s+and\s+precautions?",      "WARNINGS",           "Warnings and Precautions"),
    (r"contraindications?",                  "CONTRAINDICATIONS",  "Contraindications"),
    (r"drug\s+interactions?",                "DRUG_INTERACTIONS",  "Drug Interactions"),
    (r"dosage\s+and\s+admin",                "DOSAGE",             "Dosage and Administration"),
    (r"clinical\s+pharmacology",             "PHARMACOLOGY",       "Clinical Pharmacology"),
    (r"clinical\s+studies",                  "CLINICAL_STUDIES",   "Clinical Studies"),
    (r"overdosage",                          "OVERDOSAGE",         "Overdosage"),
    (r"how\s+supplied",                      "HOW_SUPPLIED",       "How Supplied"),
]

def detect_section(text: str):
    """Scan first 300 chars of a page for a drug-label section heading."""
    sample = text[:300].lower()
    for pattern, section_id, section_name in SECTION_MAP:
        if re.search(pattern, sample, re.MULTILINE):
            return section_id, section_name
    return None, None

# ---------------------------------------------------------------------------
# CONTENT FLAGS  — drug-specific equivalents of has_numbers / has_table
def flag_content(text: str) -> dict:
    t = text.lower()
    return {
        "has_percentages":    bool(re.search(r'\d+\.?\d*\s*%', text)),
        "has_dosage_numbers": bool(re.search(r'\d+\s*(mg|mcg|ml|g|units?)\b', t)),
        "mentions_death":     any(w in t for w in ["fatal", "death", "mortality", "died"]),
        "mentions_pregnancy": any(w in t for w in ["pregnancy", "pregnant", "fetal", "neonatal"]),
        "mentions_pediatric": any(w in t for w in ["pediatric", "children", "adolescent"]),
        "mentions_renal":     any(w in t for w in ["renal", "kidney", "creatinine"]),
        "mentions_hepatic":   any(w in t for w in ["hepatic", "liver", "hepatotoxicity"]),
        "has_table":          text.count('\n') > 8 and '\t' in text,
        "is_short_chunk":     len(text) < 200,
        "char_count":         len(text),
        "word_count":         len(text.split()),
    }

# ---------------------------------------------------------------------------
# METADATA ENRICHMENT
def enrich_metadata(doc, drug_name: str, chunk_index: int,
                    total_chunks: int, section_id: str, section_name: str):
    flags = flag_content(doc.page_content)
    doc.metadata.update({
        "drug":          drug_name,
        "section":       section_id,
        "section_name":  section_name,
        "chunk_index":   chunk_index,
        "total_chunks":  total_chunks,
        "is_first_chunk": chunk_index == 0,
        "is_last_chunk":  chunk_index == total_chunks - 1,
        "source_file":   doc.metadata.get("source", ""),
        "indexed_at":    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        **flags,
    })

# ---------------------------------------------------------------------------
# PDF LOADER
def load_pdf_with_sections(pdf_path: str, drug_name: str):
    """Load a drug-label PDF, tag pages with section metadata, split and enrich."""
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()

    current_id   = "UNKNOWN"
    current_name = "Unknown"

    for doc in pages:
        detected_id, detected_name = detect_section(doc.page_content)
        if detected_id:
            current_id   = detected_id
            current_name = detected_name
        doc.metadata["drug"]         = drug_name
        doc.metadata["section"]      = current_id
        doc.metadata["section_name"] = current_name

    print(f"  {len(pages)} pages loaded for {drug_name}.")
    splits = text_splitter.split_documents(pages)
    total  = len(splits)
    print(f"  {total} chunks created for {drug_name}.")

    for i, chunk in enumerate(splits):
        enrich_metadata(
            chunk, drug_name, i, total,
            chunk.metadata["section"],
            chunk.metadata["section_name"],
        )
    return splits

# ---------------------------------------------------------------------------
# JSON METADATA SUMMARY  (per drug, for UI display / charts)
def save_drug_metadata_json(splits: list, drug_name: str) -> None:
    sections: dict = {}
    for chunk in splits:
        s    = chunk.metadata.get("section", "UNKNOWN")
        snam = chunk.metadata.get("section_name", "Unknown")
        page = chunk.metadata.get("page", 0)
        if s not in sections:
            sections[s] = {"section_id": s, "section_name": snam,
                           "page_start": page, "page_end": page, "chunk_count": 0}
        else:
            sections[s]["page_end"] = max(sections[s]["page_end"], page)
        sections[s]["chunk_count"] += 1

    output = {
        "drug":         drug_name,
        "total_pages":  max(c.metadata.get("page", 0) for c in splits),
        "total_chunks": len(splits),
        "indexed_at":   datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sections":     list(sections.values()),
    }
    os.makedirs(METADATA_DIR, exist_ok=True)
    path = os.path.join(METADATA_DIR, f"{drug_name}_metadata.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Metadata saved → {path}")

# ---------------------------------------------------------------------------
# REGISTRY  (tracks which drugs are indexed)
def load_indexed_drugs() -> set:
    if os.path.exists(REGISTRY):
        with open(REGISTRY) as f:
            return set(json.load(f))
    return set()

def save_indexed_drugs(drugs: set) -> None:
    with open(REGISTRY, "w") as f:
        json.dump(sorted(drugs), f, indent=2)

# ---------------------------------------------------------------------------
# BOOT  — load or build the FAISS index
indexed_drugs = load_indexed_drugs()
vector_store  = None

# Collect any PDFs already sitting in PDF_DIR
os.makedirs(PDF_DIR, exist_ok=True)
existing_pdfs = [
    (os.path.join(PDF_DIR, f), os.path.splitext(f)[0].replace("_", " ").title())
    for f in os.listdir(PDF_DIR) if f.endswith(".pdf")
]

if os.path.exists(INDEX_PATH):
    print("Loading existing FDA index from disk...")
    vector_store = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    new_splits = []
    for pdf_path, drug_name in existing_pdfs:
        if drug_name not in indexed_drugs:
            print(f"New drug found, indexing {drug_name}...")
            splits = load_pdf_with_sections(pdf_path, drug_name)
            save_drug_metadata_json(splits, drug_name)
            new_splits.extend(splits)
            indexed_drugs.add(drug_name)
    if new_splits:
        vector_store.add_documents(new_splits)
        vector_store.save_local(INDEX_PATH)
        save_indexed_drugs(indexed_drugs)
        print(f"Index updated with {len(new_splits)} new chunks.")
    else:
        print("No new drugs to index.")

elif existing_pdfs:
    all_splits = []
    for pdf_path, drug_name in existing_pdfs:
        print(f"Loading {drug_name}...")
        splits = load_pdf_with_sections(pdf_path, drug_name)
        save_drug_metadata_json(splits, drug_name)
        all_splits.extend(splits)
        indexed_drugs.add(drug_name)
    if all_splits:
        print(f"Building FAISS index from {len(all_splits)} chunks...")
        vector_store = FAISS.from_documents(all_splits, embeddings)
        vector_store.save_local(INDEX_PATH)
        save_indexed_drugs(indexed_drugs)
        print("FDA index saved.")
else:
    print("No PDFs found in fda_pdfs/. Add drugs via the UI or fda_ingest.py.")

# ---------------------------------------------------------------------------
# SMART FILTER BUILDER
def build_filter(query: str, drug: str | None) -> dict:
    q = query.lower()
    base = {"drug": drug} if drug else {}

    if any(w in q for w in ["side effect", "adverse", "reaction", "toxicity", "toxic"]):
        return {**base, "section": "ADVERSE_REACTIONS", "is_short_chunk": False}
    if any(w in q for w in ["warning", "precaution", "black box", "boxed"]):
        return {**base, "section": "WARNINGS", "is_short_chunk": False}
    if any(w in q for w in ["contraindicate", "should not", "avoid", "not recommended"]):
        return {**base, "section": "CONTRAINDICATIONS", "is_short_chunk": False}
    if any(w in q for w in ["dose", "dosage", "how much", "how to take", "administration"]):
        return {**base, "section": "DOSAGE", "is_short_chunk": False}
    if any(w in q for w in ["interact", "combination", "taken with"]):
        return {**base, "section": "DRUG_INTERACTIONS", "is_short_chunk": False}
    if any(w in q for w in ["pregnant", "pregnancy", "fetal", "breastfeed"]):
        return {**base, "mentions_pregnancy": True, "is_short_chunk": False}
    if any(w in q for w in ["child", "pediatric", "kid"]):
        return {**base, "mentions_pediatric": True, "is_short_chunk": False}
    if any(w in q for w in ["overdose", "overdosage", "too much"]):
        return {**base, "section": "OVERDOSAGE", "is_short_chunk": False}
    if any(w in q for w in ["what is", "used for", "indication", "treat"]):
        return {**base, "section": "INDICATIONS", "is_short_chunk": False}
    if any(w in q for w in ["mechanism", "how does", "pharmacology", "works"]):
        return {**base, "section": "PHARMACOLOGY", "is_short_chunk": False}

    return {**base, "is_short_chunk": False} if base else {}

# ---------------------------------------------------------------------------
# DRUG DETECTION FROM QUERY
def detect_drug(query: str) -> str | None:
    q = query.lower()
    for drug in indexed_drugs:
        if drug.lower() in q:
            return drug
    return None

# ---------------------------------------------------------------------------
# MAIN ASK FUNCTION
def ask(query: str) -> str:
    if vector_store is None:
        return "No drug labels have been indexed yet. Use the 'Add Drug' tab to ingest one."

    drug        = detect_drug(query)
    meta_filter = build_filter(query, drug)

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k":      6,
            "filter": meta_filter if meta_filter else None,
        }
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
            drug_tag     = doc.metadata.get("drug", "Unknown")
            page         = doc.metadata.get("page", "?")
            section_name = doc.metadata.get("section_name", "")
            ref = (f"{drug_tag} — {section_name} (p.{page})"
                   if section_name else f"{drug_tag} (p.{page})")
            refs.add(ref)
        answer += "\n\n📄 Sources: " + " | ".join(sorted(refs))

    return answer

# ---------------------------------------------------------------------------
# SIDE EFFECTS EXTRACTION  — used by Charts tab
SIDE_EFFECTS_QUERY = """
You are analyzing the FDA drug label for {drug}.

Extract the most commonly mentioned adverse reactions / side effects and their
approximate incidence percentages (or frequency labels like "common", "rare").

Respond ONLY with a JSON object in this exact format:
{{
    "drug": "{drug}",
    "adverse_reactions": [
        {{"effect": "Headache", "frequency": "common", "pct": 15.0}},
        {{"effect": "Nausea",   "frequency": "common", "pct": 12.0}},
        ...
    ]
}}

Rules:
- Include 5–15 effects.
- "pct" should be a float (use 0 if not stated numerically).
- Use only information explicitly stated in the label.
"""

def extract_side_effects(drug_name: str) -> dict | None:
    if vector_store is None:
        return None

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k":      10,
            "filter": {"drug": drug_name, "section": "ADVERSE_REACTIONS",
                       "is_short_chunk": False},
        }
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": side_effects_template},
    )
    try:
        result = qa_chain.invoke(
            {"query": SIDE_EFFECTS_QUERY.format(drug=drug_name)}
        )
        raw   = result["result"]
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            print(f"[extract_side_effects] No JSON for {drug_name}")
            return None
        data = json.loads(match.group())
        if "adverse_reactions" not in data:
            return None
        return data
    except Exception as e:
        print(f"[extract_side_effects] Error for {drug_name}: {e}")
        return None

# ---------------------------------------------------------------------------
# DYNAMIC DRUG ADDITION  (called from app.py after PDF is downloaded)
def add_drug_from_pdf(drug_name: str, pdf_path: str) -> str:
    global vector_store

    if drug_name in indexed_drugs:
        return f"'{drug_name}' is already indexed."

    if not os.path.exists(pdf_path):
        return f"PDF not found at {pdf_path}."

    try:
        splits = load_pdf_with_sections(pdf_path, drug_name)
        save_drug_metadata_json(splits, drug_name)

        if vector_store is None:
            vector_store = FAISS.from_documents(splits, embeddings)
        else:
            vector_store.add_documents(splits)

        vector_store.save_local(INDEX_PATH)
        indexed_drugs.add(drug_name)
        save_indexed_drugs(indexed_drugs)

        return f"✅ '{drug_name}' indexed successfully ({len(splits)} chunks)."
    except Exception as e:
        return f"❌ Error indexing '{drug_name}': {e}"
