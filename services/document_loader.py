import os
import pandas as pd
from pypdf import PdfReader
from docx import Document
from app_logging import logger
# Import batch_documents alongside your existing add_document
from services.rag_service import add_document 

CHUNK_SIZE = 1000

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

def row_to_text(row_dict):
    """Convert Excel row dict to searchable natural text"""
    title = row_dict.get('title', '').strip()
    product_name = row_dict.get('product_name', '').strip()
    brand = row_dict.get('brand_name', '').strip()
    sales = row_dict.get('Sales_price', '').strip()
    repair = row_dict.get('repair_price', '').strip()
    qty = row_dict.get('qty', '').strip()
    model = row_dict.get('model_number', '').strip()
    suitable = row_dict.get('suitable_model', '').strip()
    desc = row_dict.get('product_description', '').strip()

    name = product_name or title
    if not name:
        return ""

    return (
        f"Product: {name}. "
        f"Brand: {brand}. "
        f"Sales Price: {sales}. "
        f"Repair Price: {repair}. "
        f"Stock: {qty}. "
        f"Model: {model}. "
        f"Compatible Models: {suitable}. "
        f"Description: {desc}."
    )

def read_xlsx(file_path):
    chunks = []
    try:
        excel_data = pd.read_excel(file_path, sheet_name=None, dtype=str)
        for sheet_name, df in excel_data.items():
            df = df.fillna("")
            df.columns = [str(c).strip() for c in df.columns]

            for _, row in df.iterrows():
                text_row = row_to_text(row.to_dict())
                if text_row:
                    chunks.append(text_row)
    except Exception as e:
        logger.exception("Error reading XLSX %s: %s", file_path, e)

    return "\n\n".join(chunks)

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
    """Processes a file using batch array uploads instead of disk writes."""
    try:
        status_dir = os.path.join("uploads", "processed_markers")
        os.makedirs(status_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        marker_path = os.path.join(status_dir, f"{customer_name}_{filename}.done")

        if os.path.exists(marker_path):
            return True

        logger.info("Processing upload file: %s", file_path)
        text = extract_text(file_path)
        if not text.strip():
            logger.warning("No text extracted from: %s", file_path)
            return False

        chunks = chunk_text(text)
        
        # NOTE: If your services/rag_service.py has an optimized batching endpoint, 
        # you can pass the entire 'chunks' array down at once. Otherwise, we batch write
        # groups of strings in memory without writing temporary text chunks to disk.
        success = True
        for i, chunk in enumerate(chunks):
            # Safe Fallback: We pass the chunk text dynamically down to the database wrapper
            # If your add_document only takes file paths, adjust add_document to take raw text strings.
            result = add_document(chunk, customer_name=customer_name, is_raw_text=True)
            if not result:
                success = False

        if success:
            with open(marker_path, "w", encoding="utf-8") as marker:
                marker.write("done")
            logger.info("Successfully finished processing: %s", filename)

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

    skipped_count = 0
    processed_count = 0

    for customer_name in customer_folders:
        customer_folder = os.path.join(uploads_root, customer_name)
        if customer_name == "processed_markers":
            continue
        try:
            for filename in os.listdir(customer_folder):
                file_path = os.path.join(customer_folder, filename)
                if os.path.isdir(file_path) or filename.endswith('.tmp.txt'):
                    continue
                ext = os.path.splitext(filename)[1].lower()
                
                if ext in valid_extensions:
                    marker_file = os.path.join("uploads", "processed_markers", f"{customer_name}_{filename}.done")
                    if os.path.exists(marker_file):
                        skipped_count += 1
                        continue
                        
                    process_file(file_path, customer_name)
                    processed_count += 1
        except Exception as e:
            logger.error("Error reading folder %s: %s", customer_name, e)

    # Output one clean single log summary line instead of 700 lines
    logger.info("Uploads scan done. Processed: %d new files. Skipped: %d cached files.", processed_count, skipped_count)
