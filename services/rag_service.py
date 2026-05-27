import os
import hashlib
import chromadb

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


from app_logging import logger

# -----------------------------
# CONFIG
# -----------------------------

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "business_docs"

# -----------------------------
# INIT EMBEDDING MODEL
# -----------------------------

logger.info("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# INIT CHROMADB
# -----------------------------

logger.info("Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)

logger.info("ChromaDB initialized successfully")


# -----------------------------
# TEXT SPLITTER
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)


# -----------------------------
# GENERATE FILE HASH
# -----------------------------

def generate_file_hash(file_path):
    """
    Generate SHA256 hash for duplicate prevention
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            sha256.update(chunk)

    return sha256.hexdigest()


# -----------------------------
# LOAD FILE CONTENT
# -----------------------------

def load_text_file(file_path):
    """
    Load .md or .txt files
    """

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# CHUNK DOCUMENT
# -----------------------------

def split_document(text):
    """
    Split large document into chunks
    """

    chunks = splitter.split_text(text)

    cleaned = [c.strip() for c in chunks if c.strip()]

    return cleaned


# -----------------------------
# ADD DOCUMENT TO CHROMADB
# -----------------------------

def add_document(
    file_path,
    customer_name="default_customer"
):
    """
    Load, chunk, embed, and store document with dual-layer duplication blocking 
    (ChromaDB index check + local container disk marker check).
    """
    try:
        logger.info("Processing file: %s", file_path)

        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            return False

        # Extract the true source filename (stripping out any temporary extension noise)
        base_filename = os.path.basename(file_path)
        if base_filename.endswith(".tmp.txt"):
            source_identity = base_filename.replace(".tmp.txt", "")
        else:
            source_identity = base_filename

        # -------------------------------------------------------------
        # LAYER 1: DISK MARKER CHECK (Fast short-circuit for container reboots)
        # -------------------------------------------------------------
        marker_file = file_path + ".done"
        if os.path.exists(marker_file):
            logger.info("--> [FAST SKIP] File already processed in this runtime lifecycle: %s", base_filename)
            return True

        # -------------------------------------------------------------
        # LAYER 2: VECTOR DB DUPLICATE CHECK BY SOURCE FILENAME
        # -------------------------------------------------------------
        existing = collection.get(
            where={"source": source_identity}
        )

        if existing and existing.get("ids") and len(existing["ids"]) > 0:
            logger.info(
                "Skipping duplicate file (already indexed in vector DB): %s",
                source_identity
            )
            # Create the disk marker so layer 1 catches it next time without hitting the DB
            with open(marker_file, "w", encoding="utf-8") as marker:
                marker.write("done")
            return True

        file_hash = generate_file_hash(file_path)

        # -----------------------------
        # LOAD CONTENT
        # -----------------------------
        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".md", ".txt"]:
            text = load_text_file(file_path)
        else:
            logger.warning(
                "Unsupported file type: %s",
                ext
            )
            return False

        if not text.strip():
            logger.warning("Empty document")
            return False

        # -----------------------------
        # SPLIT INTO CHUNKS
        # -----------------------------
        chunks = split_document(text)

        logger.info(
            "Generated %d chunks",
            len(chunks)
        )

        # -----------------------------
        # CREATE EMBEDDINGS
        # -----------------------------
        embeddings = embed_model.encode(chunks).tolist()

        # -----------------------------
        # CREATE UNIQUE IDS
        # -----------------------------
        ids = [
            f"{customer_name}_{source_identity}_{i}"
            for i in range(len(chunks))
        ]

        # -----------------------------
        # METADATA MAPPING
        # -----------------------------
        metadatas = []
        for i in range(len(chunks)):
            metadatas.append({
                "source": source_identity,
                "customer": customer_name,
                "file_hash": file_hash,
                "chunk_index": i
            })

        # -----------------------------
        # STORE IN CHROMADB
        # -----------------------------
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        # -----------------------------
        # WRITE SUCCESS MARKER TO DISK
        # -----------------------------
        with open(marker_file, "w", encoding="utf-8") as marker:
            marker.write("done")

        logger.info(
            "Successfully added document to Vector DB: %s",
            source_identity
        )
        return True

    except Exception as e:
        logger.exception(
            "Error adding document: %s",
            file_path
        )
        return False




# -----------------------------
# SEARCH DOCUMENTS
# -----------------------------

def search_documents(
    query,
    customer_name="default_customer",
    n_results=15
):
    """
    Completely generic, brand-agnostic hybrid scanner.
    Works for any phone, model number, or component variation automatically.
    """
    documents = []
    query_clean = query.lower()

    # -------------------------------------------------------------
    # LAYER 1: DYNAMIC KEYWORD INTERSECTION FILTER
    # -------------------------------------------------------------
    try:
        import re
        # Isolate alphanumeric clusters (e.g. "iPhone 7 LCD" -> ["iphone", "7", "lcd"])
        raw_tokens = re.findall(r'\b\w+\b', query_clean)
        
        # Comprehensive language noise baseline filter
        filler_words = {
            "what", "is", "the", "of", "price", "details", "for", "in", 
            "stock", "enquiry", "tell", "me", "pls", "please", "show", 
            "check", "cost", "how", "much", "find", "list", "get", "with", "any"
        }
        
        # Extract only the critical unique keywords typed by the client
        search_terms = [w for w in raw_tokens if w not in filler_words and len(w) > 0]

        fallback_file = f"uploads/{customer_name}/Product_ohms.xlsx.tmp.txt"
        if os.path.exists(fallback_file) and search_terms:
            with open(fallback_file, "r", encoding="utf-8") as f:
                all_lines = [line.strip() for line in f.readlines() if line.strip()]

            # 1. Match rows that intersect with ALL user-provided identifiers
            for line in all_lines:
                line_lower = line.lower()
                if all(term in line_lower for term in search_terms):
                    if line not in documents:
                        documents.append(line)
                if len(documents) >= 6:
                    break

            # 2. Match rows intersecting with MOST identifiers if an exact match is empty
            if not documents:
                # Require at least 2 identifiers to match to avoid irrelevant rows
                required_match_count = max(2, len(search_terms) - 1) if len(search_terms) > 1 else 1
                for line in all_lines:
                    line_lower = line.lower()
                    matches = sum(1 for term in search_terms if term in line_lower)
                    if matches >= required_match_count:
                        if line not in documents:
                            documents.append(line)
                    if len(documents) >= 5:
                        break
                        
    except Exception as fallback_err:
        logger.error("Dynamic keyword parser failed: %s", fallback_err)

    # -------------------------------------------------------------
    # LAYER 2: CHROMADB VECTOR BACKFILL
    # -------------------------------------------------------------
    try:
        query_embedding = embed_model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        if results and results.get("documents") and results["documents"]:
            # Safely loop through and flatten ChromaDB's matrix array block
            target_pool = results["documents"][0] if isinstance(results["documents"][0], list) else results["documents"]
            for doc in target_pool:
                if doc and str(doc) not in documents:
                    documents.append(str(doc))
                    
    except Exception as e:
        logger.exception("Error searching semantic documents: %s", e)

    return documents[:n_results]





# -----------------------------
# BUILD CONTEXT
# -----------------------------

def build_context(documents):
    """
    Join retrieved chunks into context
    """

    if not documents:
        return "No relevant information found."

    return "\n\n".join(documents)


# -----------------------------
# AUTO LOAD FAQ
# -----------------------------

def initialize_default_documents():

    faq_path = "faq.md"

    if os.path.exists(faq_path):

        logger.info(
            "Initializing default FAQ document..."
        )

        add_document(
            faq_path,
            customer_name="beesbuzz"
        )

    else:
        logger.warning(
            "faq.md not found. Skipping preload."
        )


# -----------------------------
# INIT ON STARTUP
# -----------------------------

initialize_default_documents()
