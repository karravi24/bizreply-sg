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
    Load, chunk, embed, and store document
    """

    try:

        logger.info("Processing file: %s", file_path)

        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            return False

        file_hash = generate_file_hash(file_path)

        # -----------------------------
        # CHECK DUPLICATE
        # -----------------------------

        existing = collection.get(
            where={"file_hash": file_hash}
        )

        if existing and existing.get("ids"):
            logger.info(
                "Skipping duplicate file: %s",
                file_path
            )
            return True

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

        embeddings = embed_model.encode(
            chunks
        ).tolist()

        # -----------------------------
        # CREATE IDS
        # -----------------------------

        ids = [
            f"{file_hash}_{i}"
            for i in range(len(chunks))
        ]

        # -----------------------------
        # METADATA
        # -----------------------------

        metadatas = []

        for i in range(len(chunks)):
            metadatas.append({
                "source": os.path.basename(file_path),
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

        logger.info(
            "Successfully added document: %s",
            file_path
        )

        return True

    except Exception as e:
        logger.exception(
            "Error adding document: %s",
            e
        )
        return False


# -----------------------------
# SEARCH DOCUMENTS
# -----------------------------

def search_documents(
    query,
    customer_name="default_customer",
    n_results=3
):
    """
    Semantic search from ChromaDB
    """

    try:

        query_embedding = embed_model.encode(
            [query]
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where={
                "customer": customer_name
            }
        )

        documents = []

        if (
            results
            and results.get("documents")
            and results["documents"][0]
        ):
            documents = results["documents"][0]

        return documents

    except Exception as e:
        logger.exception(
            "Error searching documents: %s",
            e
        )

        return []


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