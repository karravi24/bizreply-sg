import os
import glob

from pypdf import PdfReader
from docx import Document

from app_logging import logger
import pandas as pd
from services.rag_service import add_document


# -----------------------------------
# READ PDF
# -----------------------------------

def read_pdf(file_path):

    text = ""

    try:

        reader = PdfReader(file_path)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception as e:
        logger.exception(
            "Error reading PDF: %s",
            e
        )

    return text


# -----------------------------------
# READ DOCX
# -----------------------------------

def read_docx(file_path):

    text = ""

    try:

        doc = Document(file_path)

        for para in doc.paragraphs:

            if para.text.strip():
                text += para.text + "\n"

    except Exception as e:
        logger.exception(
            "Error reading DOCX: %s",
            e
        )

    return text

# -----------------------------------
# READ XLSX
# -----------------------------------

def read_xlsx(file_path):

    text = ""

    try:

        excel_data = pd.read_excel(
            file_path,
            sheet_name=None
        )

        for sheet_name, df in excel_data.items():

            text += f"\nSheet: {sheet_name}\n"

            df = df.fillna("")

            for row in df.values:

                row_text = " | ".join(
                    [str(cell) for cell in row]
                )

                text += row_text + "\n"

    except Exception as e:

        logger.exception(
            "Error reading XLSX: %s",
            e
        )

    return text

# -----------------------------------
# EXTRACT TEXT FROM FILE
# -----------------------------------

def extract_text(file_path):

    ext = os.path.splitext(file_path)[1].lower()

    # -------------------------
    # TXT / MD
    # -------------------------

    if ext in [".txt", ".md"]:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            return f.read()

    # -------------------------
    # PDF
    # -------------------------

    elif ext == ".pdf":

        return read_pdf(file_path)

    # -------------------------
    # DOCX
    # -------------------------

    elif ext == ".docx":

        return read_docx(file_path)

    # -------------------------
    # XLSX
    # -------------------------
    elif ext == ".xlsx":

        return read_xlsx(file_path)

    # -------------------------
    # UNSUPPORTED
    # -------------------------

    else:

        logger.warning(
            "Unsupported file type: %s",
            ext
        )

        return ""


# -----------------------------------
# PROCESS SINGLE FILE
# -----------------------------------

def process_file(
    file_path,
    customer_name
):

    try:

        logger.info(
            "Processing upload file: %s",
            file_path
        )

        text = extract_text(file_path)

        if not text.strip():

            logger.warning(
                "No text extracted from: %s",
                file_path
            )

            return False

        # -------------------------
        # TEMP TEXT FILE
        # -------------------------

        temp_path = file_path + ".tmp.txt"

        with open(
            temp_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(text)

        # -------------------------
        # ADD TO VECTOR DB
        # -------------------------

        result = add_document(
            temp_path,
            customer_name=customer_name
        )

        # -------------------------
        # DELETE TEMP FILE
        # -------------------------

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return result

    except Exception as e:

        logger.exception(
            "Error processing file: %s",
            e
        )

        return False


# -----------------------------------
# SCAN UPLOADS FOLDER
# -----------------------------------

def scan_uploads_folder():

    uploads_root = "uploads"

    if not os.path.exists(uploads_root):

        logger.warning(
            "uploads folder not found"
        )

        return

    logger.info(
        "Scanning uploads folder..."
    )

    # -----------------------------------
    # LOOP CUSTOMER FOLDERS
    # -----------------------------------

    customer_folders = glob.glob(
        os.path.join(
            uploads_root,
            "*"
        )
    )

    for customer_folder in customer_folders:

        if not os.path.isdir(customer_folder):
            continue

        customer_name = os.path.basename(
            customer_folder
        )

        logger.info(
            "Customer folder: %s",
            customer_name
        )

        # -----------------------------------
        # FIND FILES
        # -----------------------------------

        patterns = [
            "*.pdf",
            "*.docx",
            "*.xlsx",
            "*.txt",
            "*.md"
        ]

        all_files = []

        for pattern in patterns:

            all_files.extend(
                glob.glob(
                    os.path.join(
                        customer_folder,
                        pattern
                    )
                )
            )

        # -----------------------------------
        # PROCESS FILES
        # -----------------------------------

        for file_path in all_files:

            process_file(
                file_path,
                customer_name
            )

    logger.info(
        "Uploads folder scan completed"
    )