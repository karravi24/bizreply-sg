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
COLLECTION_NAME = "business_docs"
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

        chunks = split_document(text)
        logger.info("Generated %d chunks for %s", len(chunks), source_identity)

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

    # Keyword fallback only if vector search fails
    try:
        import re
        tokens = re.findall(r'\b\w+\b', query_clean)
        filler_words = {"what", "is", "the", "of", "price", "details", "for", "in", "stock", "enquiry", "tell", "me", "pls", "please", "show", "check", "cost", "how", "much", "find", "list", "get", "with", "any"}
        search_terms = [w for w in tokens if w not in filler_words and len(w) > 1]

        if search_terms:
            results = collection.get(where={"customer": customer_name})
            if results and results.get("documents"):
                for doc in results["documents"]:
                    doc_lower = doc.lower()
                    if all(term in doc_lower for term in search_terms):
                        documents.append(doc)
                    if len(documents) >= 6:
                        break
    except Exception as e:
        logger.error("Keyword fallback failed: %s", e)

    # Vector search
    try:
        query_embedding = embed_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=n_results, where={"customer": customer_name})
        if results and results.get("documents"):
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
