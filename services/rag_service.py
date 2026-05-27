import os
import hashlib
import re
import chromadb
import pandas as pd
from pypdf import PdfReader
from docx import Document
from chromadb.utils.embedding_functions import GeminiEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app_logging import logger

# -----------------------------
# CONFIG
# -----------------------------
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "business_docs_v3"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Global Init for Cloud-Based Gemini Embeddings (Fixes CPU Bottleneck)
GEMINI_KEY = os.getenv("GEMINI_KEY")
if not GEMINI_KEY:
    logger.error("CRITICAL: GEMINI_KEY is missing from environment variables!")
    raise ValueError("GEMINI_KEY must be provided for cloud embedding functions.")

logger.info("Initializing Gemini Cloud Embedding Engine...")
gemini_ef = GeminiEmbeddingFunction(
    api_key=GEMINI_KEY,
    model_name="models/text-embedding-004"
)

logger.info("Initializing ChromaDB Index Storage...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=gemini_ef # 👈 Shifts calculation stress away from Railway CPU
)
logger.info("ChromaDB vector matrix connected successfully.")

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
    if not os.path.exists(file_path):
        return "raw_text_no_hash"
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
def add_document(input_source, customer_name="default_customer", is_raw_text=False):
    """
    Saves and indexes assets. Supports file paths or raw text strings.
    """
    try:
        if not is_raw_text:
            if not os.path.exists(input_source):
                logger.error("File not found: %s", input_source)
                return False
            base_filename = os.path.basename(input_source)
            source_identity = base_filename.replace(".tmp.txt", "") if base_filename.endswith(".tmp.txt") else base_filename
            text = extract_text(input_source)
            file_hash = generate_file_hash(input_source)
        else:
            # Handle in-memory streaming text blocks directly
            source_identity = f"raw_stream_{hashlib.md5(input_source.encode()).hexdigest()[:8]}"
            text = input_source
            file_hash = "in_memory_stream"

        if not text.strip():
            return False

        # Parse data row boundaries safely
        if "Product:" in text or source_identity.endswith(('.xlsx', '.xls', '.csv')):
            raw_lines = text.split("\n")
            chunks = [line.strip() for line in raw_lines if line.strip() and "Product:" in line]
        else:
            chunks = split_document(text)

        if not chunks:
            return False

        # Bulk generation configuration setup
        ids = [f"{customer_name}_{source_identity}_{i}" for i in range(len(chunks))]
        metadatas = [{
            "source": source_identity, 
            "customer": customer_name, 
            "file_hash": file_hash, 
            "chunk_index": i
        } for i in range(len(chunks))]

        # ChromaDB automatically vectorises via gemini_ef in the cloud now!
        collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        return True

    except Exception as e:
        logger.exception("Error writing data payload to index database layer: %s", e)
        return False

# -----------------------------
# SEARCH DOCUMENTS
# -----------------------------
def search_documents(query, customer_name="default_customer", n_results=15):
    documents = []
    query_clean = query.lower()

    try:
        tokens = re.findall(r'\b\w+\b', query_clean)
        filler_words = {"what", "is", "the", "of", "price", "details", "for", "in", "stock", "enquiry", "tell", "me", "pls", "please", "show", "check", "cost", "how", "much", "find", "list", "get", "with", "any"}
        search_terms = [w for w in tokens if w not in filler_words and len(w) > 0]

        logger.info("Keyword search terms parsed: %s", search_terms)

        if search_terms:
            results = collection.get(where={"customer": customer_name})
            if results and results.get("documents"):
                all_docs = results["documents"]

                strict_matches = []
                for doc in all_docs:
                    doc_lower = doc.lower()
                    if all(re.search(r'\b' + re.escape(term) + r'\b', doc_lower) for term in search_terms):
                        strict_matches.append(doc)
                    if len(strict_matches) >= 6:
                        break

                if strict_matches:
                    logger.info("Found %d strict keyword matches", len(strict_matches))
                    documents.extend(strict_matches)
                else:
                    # Generic partial term match routing fallback
                    min_matches = 2 if len(search_terms) > 2 else 1
                    for doc in all_docs:
                        doc_lower = doc.lower()
                        match_count = sum(1 for term in search_terms if re.search(r'\b' + re.escape(term) + r'\b', doc_lower))
                        if match_count >= min_matches:
                            documents.append(doc)
                        if len(documents) >= n_results:
                            break

        # If keyword search returned nothing, fall back to structural cloud vector searches
        if not documents:
            logger.info("Falling back to vector semantic embedding lookup.")
            # Chroma DB uses the cloud gemini_ef directly for query embedding compilation
            vector_results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"customer": customer_name}
            )
            if vector_results and vector_results.get("documents"):
                documents = vector_results["documents"][0]

    except Exception as e:
        logger.exception("Error executing search queries across collection matrix: %s", e)
        
    return documents

def build_context(documents):
    if not documents:
        return "No relevant information found."
    return "\n".join(documents)


def initialize_default_documents():
    faq_path = "faq.md"
    if os.path.exists(faq_path):
        logger.info("Initializing default FAQ document...")
        add_document(faq_path, customer_name="beesbuzz")
    else:
        logger.warning("faq.md not found. Skipping preload.")

initialize_default_documents()
