import os
import pandas as pd
from pypdf import PdfReader
from docx import Document
from app_logging import logger
from services.rag_service import add_document

CHUNK_SIZE = 1000 # characters per chunk

def chunk_text(text, size=CHUNK_SIZE):
    """Split text into chunks by paragraphs, not hard cut."""
    chunks = []
    buffer = ""
    for para in text.split("\n\n"):
        if len(buffer) + len(para) > size and buffer:
            chunks.append(buffer.strip())
            buffer = para
        else:
            buffer += "\n\n" + para
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks

def read_pdf(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
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

def read_xlsx(file_path):
    text = ""
    try:
        excel_data = pd.read_excel(file_path, sheet_name=None, dtype=str)
        for sheet_name, df in excel_data.items():
            df = df.fillna("")
            text += f"\nSheet: {sheet_name}\n"
            for _, row in df.iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if val])
                if row_text:
                    text += row_text + "\n"
    except Exception as e:
        logger.exception("Error reading XLSX %s: %s", file_path, e)
    return text

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".txt", ".md"]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.exception("Error reading TXT %s: %s", file_path, e)
            return ""
    elif ext == ".pdf":
        return read_pdf(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    elif ext == ".xlsx":
        return read_xlsx(file_path)
    else:
        logger.warning("Unsupported file type: %s", ext)
        return ""

def process_file(file_path, customer_name):
    try:
        logger.info("Processing upload file: %s", file_path)

        status_dir = os.path.join("uploads", "processed_markers")
        os.makedirs(status_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        marker_path = os.path.join(status_dir, f"{customer_name}_{filename}.done")

        if os.path.exists(marker_path):
            logger.info("--> [FAST SKIP] Already processed: %s", filename)
            return True

        text = extract_text(file_path)
        if not text.strip():
            logger.warning("No text extracted from: %s", file_path)
            return False

        chunks = chunk_text(text)
        success = True

        for i, chunk in enumerate(chunks):
            temp_path = f"{file_path}.chunk_{i}.tmp.txt"
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(chunk)

            result = add_document(temp_path, customer_name=customer_name)
            if not result:
                success = False

            # Delete temp unless it's Excel and you want to keep it
            if not (file_path.endswith(('.xlsx', '.xls')) and os.path.exists(temp_path)):
                os.remove(temp_path)

        if success:
            with open(marker_path, "w", encoding="utf-8") as marker:
                marker.write("done")

        return success

    except Exception as e:
        logger.exception("Error processing file: %s", file_path)
        return False

def scan_uploads_folder():
    uploads_root = "uploads"
    if not os.path.exists(uploads_root):
        logger.warning("uploads folder not found")
        return

    logger.info("Scanning uploads folder...")
    valid_extensions = {".pdf", ".docx", ".xlsx", ".txt", ".md"}

    try:
        customer_folders = [f for f in os.listdir(uploads_root)
                            if os.path.isdir(os.path.join(uploads_root, f))]
    except Exception as e:
        logger.error("Failed to read uploads directory: %s", e)
        return

    for customer_name in customer_folders:
        customer_folder = os.path.join(uploads_root, customer_name)
        try:
            for filename in os.listdir(customer_folder):
                file_path = os.path.join(customer_folder, filename)
                if os.path.isdir(file_path) or filename.endswith('.tmp.txt'):
                    continue
                ext = os.path.splitext(filename)[1].lower()
                if ext in valid_extensions:
                    process_file(file_path, customer_name)
        except Exception as e:
            logger.error("Error reading folder %s: %s", customer_name, e)

    logger.info("Uploads folder scan completed")
