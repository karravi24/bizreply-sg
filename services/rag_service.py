import os
import hashlib
import chromadb
import pandas as pd
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app_logging import logger

# -----------------------------
# CONFIG
# -----------------------------
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "business_docs_v3"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

logger.info("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

logger.info("Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
logger.info("ChromaDB initialized successfully")

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# -----------------------------
# FILE READERS
# -----------------------------
def read_txt_md(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    except Exception as e:
        logger.exception("Error reading PDF %s: %s", file_path, e)
    return text

def read_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    except Exception as e:
        logger.exception("Error reading DOCX %s: %s", file_path, e)
    return text

def read_excel(file_path):
    text = ""
    try:
        df_dict = pd.read_excel(file_path, sheet_name=None, dtype=str)
        for sheet_name, df in df_dict.items():
            df = df.fillna("")
            text += f"\nSheet: {sheet_name}\n"
            for _, row in df.iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if val])
                if row_text:
                    text += row_text + "\n"
    except Exception as e:
        logger.exception("Error reading Excel %s: %s", file_path, e)
    return text

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".txt", ".md"]:
        return read_txt_md(file_path)
    elif ext == ".pdf":
        return read_pdf(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    elif ext in [".xlsx", ".xls", ".csv"]:
        return read_excel(file_path)
    else:
        logger.warning("Unsupported file type: %s", ext)
        return ""

# -----------------------------
# UTILS
# -----------------------------
def generate_file_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            sha256.update(chunk)
    return sha256.hexdigest()

def split_document(text):
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]

# -----------------------------
# ADD DOCUMENT TO CHROMADB
# -----------------------------
def add_document(file_path, customer_name="default_customer"):
    try:
        logger.info("Processing file: %s", file_path)
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            return False

        base_filename = os.path.basename(file_path)
        source_identity = base_filename.replace(".tmp.txt", "") if base_filename.endswith(".tmp.txt") else base_filename

        marker_file = file_path + ".done"
        if os.path.exists(marker_file):
            logger.info("--> [FAST SKIP] Already processed: %s", base_filename)
            return True

        existing = collection.get(where={"source": source_identity})
        if existing and existing.get("ids"):
            logger.info("Skipping duplicate file: %s", source_identity)
            with open(marker_file, "w", encoding="utf-8") as marker:
                marker.write("done")
            return True

        text = extract_text(file_path)
        if not text.strip():
            logger.warning("Empty document: %s", file_path)
            return False

        # -------------------------------------------------------------
        # ABSOLUTE FIX: FORCE STRICT ROW ISOLATION FOR SPREADSHEETS
        # -------------------------------------------------------------
        # Check if the text content looks like Excel data matrix rows
        if "Product:" in text or source_identity.endswith(('.xlsx', '.xls', '.csv')):
            # Split clean single text lines by ignoring table headers and structural sheet tags
            raw_lines = text.split("\n")
            chunks = []
            for line in raw_lines:
                clean_line = line.strip()
                # Check for explicit data flags and throw away unpopulated structures
                if clean_line and "Product:" in clean_line:
                    chunks.append(clean_line)
            logger.info("Excel Processing: Split text cleanly into %d single product rows.", len(chunks))
        else:
            # Fallback to general semantic chunks for standard documents and text assets
            chunks = split_document(text)
            logger.info("Generated %d generic chunks for text document: %s", len(chunks), source_identity)

        if not chunks:
            logger.warning("No discrete chunks generated for: %s", file_path)
            return False

        embeddings = embed_model.encode(chunks).tolist()
        ids = [f"{customer_name}_{source_identity}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source_identity, "customer": customer_name, "file_hash": generate_file_hash(file_path), "chunk_index": i} for i in range(len(chunks))]

        collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)

        with open(marker_file, "w", encoding="utf-8") as marker:
            marker.write("done")

        logger.info("Successfully added document: %s", source_identity)
        return True

    except Exception as e:
        logger.exception("Error adding document: %s", file_path)
        return False


# -----------------------------
# SEARCH DOCUMENTS
# -----------------------------
def search_documents(query, customer_name="default_customer", n_results=15):
    documents = []
    query_clean = query.lower()

    # Keyword fallback with strict word-boundary matching
    try:
        import re
        tokens = re.findall(r'\b\w+\b', query_clean)
        filler_words = {"what", "is", "the", "of", "price", "details", "for", "in", "stock", "enquiry", "tell", "me", "pls", "please", "show", "check", "cost", "how", "much", "find", "list", "get", "with", "any"}
        search_terms = [w for w in tokens if w not in filler_words and len(w) > 0] # Changed len > 0 to catch single numbers like '7'

        logger.info("Keyword search terms: %s", search_terms)

        if search_terms:
            results = collection.get(where={"customer": customer_name})
            if results and results.get("documents"):
                all_docs = results["documents"]

                # First pass: match ALL terms as whole distinct words
                strict_matches = []
                for doc in all_docs:
                    doc_lower = doc.lower()
                    # \b ensures complete standalone word matching (skips substrings like A1387)
                    if all(re.search(r'\b' + re.escape(term) + r'\b', doc_lower) for term in search_terms):
                        strict_matches.append(doc)
                    if len(strict_matches) >= 6:
                        break

                if strict_matches:
                    logger.info("Found %d strict matches for terms %s", len(strict_matches), search_terms)
                    documents.extend(strict_matches)
                else:
                    # Fallback: match at least 2 terms, or 1 if query is short (using word boundaries)
                    min_matches = 2 if len(search_terms) > 2 else 1
                    for doc in all_docs:
                        doc_lower = doc.lower()
                        match_count = sum(1 for term in search_terms if re.search(r'\b' + re.escape(term) + r'\b', doc_lower))
                        if match_count >= min_matches:
                            documents.append(doc)
                        if len(documents) >= 6:
                            break
                    logger.info("No strict match. Using fallback with min %d matches, got %d docs", min_matches, len(documents))
    except Exception as e:
        logger.error("Keyword fallback failed: %s", e)

    # Vector search - backfill and deduplicate remaining positions
    try:
        query_embedding = embed_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=n_results, where={"customer": customer_name})
        if results and results.get("documents") and results["documents"][0]:
            for doc in results["documents"][0]:
                if doc and doc not in documents:
                    documents.append(doc)
    except Exception as e:
        logger.exception("Error searching semantic documents: %s", e)

    return documents[:n_results]


def build_context(documents):
    return "\n\n".join(documents) if documents else "No relevant information found."

def initialize_default_documents():
    faq_path = "faq.md"
    if os.path.exists(faq_path):
        logger.info("Initializing default FAQ document...")
        add_document(faq_path, customer_name="beesbuzz")
    else:
        logger.warning("faq.md not found. Skipping preload.")

initialize_default_documents()