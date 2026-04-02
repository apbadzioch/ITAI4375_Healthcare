"""
fda_ingest.py  —  Download drug-label PDFs from openFDA
Usage:
    python fda_ingest.py "Aspirin"
    python fda_ingest.py "Metformin" "Lisinopril" "Atorvastatin"

The script:
  1. Hits the openFDA drug/label.json endpoint to find the SPL set_id
  2. Pulls the PDF from the DailyMed API (NIH) — the canonical source for
     FDA-approved label PDFs
  3. Saves to fda_pdfs/<drug_name>.pdf

Why DailyMed + openFDA together?
  openFDA has the search index; DailyMed has the actual PDF downloads.
"""

import os, sys, re, json, requests
from pathlib import Path

PDF_DIR = "fda_pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

OPENFDA_URL  = "https://api.fda.gov/drug/label.json"
DAILYMED_PDF = "https://dailymed.nlm.nih.gov/dailymed/downloadpdffile.cfm?setId={set_id}"
DAILYMED_SPL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name={name}&pagesize=1"


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name.strip().lower())


def search_openfda(drug_name: str) -> dict | None:
    """Return the first matching label JSON from openFDA."""
    try:
        resp = requests.get(
            OPENFDA_URL,
            params={"search": f'openfda.brand_name:"{drug_name}"', "limit": 1},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            return results[0]
        # fallback: generic name
        resp = requests.get(
            OPENFDA_URL,
            params={"search": f'openfda.generic_name:"{drug_name}"', "limit": 1},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return results[0] if results else None
    except Exception as e:
        print(f"  [openFDA] Search failed for '{drug_name}': {e}")
        return None


def get_set_id_from_dailymed(drug_name: str) -> str | None:
    """Look up the SPL set_id on DailyMed."""
    try:
        url  = DAILYMED_SPL.format(name=requests.utils.quote(drug_name))
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        spls = data.get("data", [])
        if spls:
            return spls[0].get("setid")
    except Exception as e:
        print(f"  [DailyMed] SPL lookup failed for '{drug_name}': {e}")
    return None


def download_pdf(set_id: str, out_path: str) -> bool:
    """Download the PDF from DailyMed for a given set_id."""
    url = DAILYMED_PDF.format(set_id=set_id)
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower() and "octet" not in content_type.lower():
            print(f"  [DailyMed] Unexpected content-type: {content_type}")
            return False
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  ✅ Saved {size_kb:.0f} KB → {out_path}")
        return True
    except Exception as e:
        print(f"  [DailyMed] PDF download failed: {e}")
        return False


def ingest_drug(drug_name: str) -> tuple[bool, str]:
    """
    Full pipeline: search → find set_id → download PDF.
    Returns (success: bool, pdf_path: str).
    """
    fname    = safe_filename(drug_name)
    out_path = os.path.join(PDF_DIR, f"{fname}.pdf")

    if os.path.exists(out_path):
        print(f"  PDF already exists for '{drug_name}' — skipping download.")
        return True, out_path

    print(f"\n🔍 Searching for '{drug_name}'...")

    # Try DailyMed first (better PDF availability)
    set_id = get_set_id_from_dailymed(drug_name)

    # Fall back to openFDA set_id
    if not set_id:
        label = search_openfda(drug_name)
        if label:
            set_id = label.get("set_id") or label.get("openfda", {}).get("spl_set_id", [None])[0]

    if not set_id:
        msg = f"Could not find a set_id for '{drug_name}'. Check the name and try again."
        print(f"  ❌ {msg}")
        return False, msg

    print(f"  Found set_id: {set_id}")
    success = download_pdf(set_id, out_path)
    return (success, out_path) if success else (False, f"Download failed for set_id {set_id}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    drugs = sys.argv[1:] if len(sys.argv) > 1 else ["Aspirin"]
    for drug in drugs:
        ok, result = ingest_drug(drug)
        if ok:
            print(f"  Ready for indexing: {result}")
        else:
            print(f"  Failed: {result}")
